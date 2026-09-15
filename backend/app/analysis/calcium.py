"""Calcium-imaging transient analysis.

Extracts a mean-intensity trace (whole frame, or a supplied ROI mask),
optionally subtracts a background-fluorescence trace, converts it to a
normalized dF/F0 trace, then detects individual calcium transients and
characterizes each one: amplitude, start-to-peak (Ca2+ release), peak-to-
end (reuptake), total duration, widths at 50 % / 90 % (CTD50 / CTD90), rise
time, exponential decay time constant and time-to-decay 10/50/90 %. Also
produces a pixel-wise peak dF/F0 map of the field of view.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from app.analysis.signal_common import (
    detect_peaks,
    find_local_min_between,
    safe_mean,
    smooth,
    time_to_decay,
)


@dataclass
class CalciumResult:
    fps: float
    n_frames: int
    time_s: np.ndarray
    raw_trace: np.ndarray
    df_f0: np.ndarray
    peak_indices: np.ndarray
    transients_df: pd.DataFrame
    summary: dict = field(default_factory=dict)
    background_trace: np.ndarray | None = None
    df_f0_map: np.ndarray | None = None


def _exp_decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def extract_intensity_trace(frames: np.ndarray, roi_mask: np.ndarray | None = None) -> np.ndarray:
    """Mean pixel intensity per frame, optionally restricted to an ROI mask."""
    if frames.ndim != 3:
        raise ValueError("Expected grayscale frames with shape (N, H, W)")
    if roi_mask is not None:
        if roi_mask.shape != frames.shape[1:]:
            raise ValueError("roi_mask shape must match frame shape (H, W)")
        return frames[:, roi_mask].mean(axis=1).astype(np.float64)
    return frames.mean(axis=(1, 2)).astype(np.float64)


def auto_background_trace(frames: np.ndarray, darkest_fraction: float = 0.05) -> np.ndarray:
    """Per-frame mean of the pixels that are darkest over the whole recording.

    Approximates the manual "measure a cell-free region" background step
    (as in the lab's MATLAB pipeline) without user input: pixels whose
    maximum intensity over time is in the lowest `darkest_fraction` are
    taken to be cell-free background. Works when the field of view has
    genuinely empty regions; on a fully confluent field it will instead
    pick the dimmest cells, so it's opt-in rather than default.
    """
    if frames.ndim != 3:
        raise ValueError("Expected grayscale frames with shape (N, H, W)")
    max_projection = frames.max(axis=0)
    threshold = np.percentile(max_projection, darkest_fraction * 100.0)
    mask = max_projection <= threshold
    if not mask.any():
        mask = max_projection == max_projection.min()
    return frames[:, mask].mean(axis=1).astype(np.float64)


def compute_df_f0(trace: np.ndarray, baseline_percentile: float = 10.0) -> tuple[np.ndarray, float]:
    f0 = float(np.percentile(trace, baseline_percentile))
    f0 = f0 if f0 > 1e-6 else 1e-6
    return (trace - f0) / f0, f0


def compute_pixel_df_f0_map(frames: np.ndarray, baseline_percentile: float = 10.0, max_frames: int = 200) -> np.ndarray:
    """Pixel-wise peak dF/F0 over the recording — where the calcium signal is.

    F0 per pixel is a low percentile over time (same definition as the
    trace), computed on a temporally subsampled stack (<= max_frames
    frames, float32) so a long high-resolution video doesn't need a
    full float copy in memory; the peak uses every frame.
    """
    if frames.ndim != 3:
        raise ValueError("Expected grayscale frames with shape (N, H, W)")
    step = max(1, int(np.ceil(frames.shape[0] / max_frames)))
    f0 = np.percentile(frames[::step].astype(np.float32), baseline_percentile, axis=0)
    f0 = np.maximum(f0, 1e-6)
    peak = frames.max(axis=0).astype(np.float32)
    return ((peak - f0) / f0).astype(np.float32)


def _rise_crossing_time(
    time_s: np.ndarray, values: np.ndarray, onset_idx: int, peak_idx: int, level: float
) -> float | None:
    """Time (absolute) at which the rising phase first reaches `level`, interpolated."""
    for k in range(onset_idx, peak_idx):
        v0, v1 = values[k], values[k + 1]
        if v1 >= level:
            if v1 == v0:
                return float(time_s[k])
            frac = (level - v0) / (v1 - v0)
            return float(time_s[k] + frac * (time_s[k + 1] - time_s[k]))
    return None


def _width_at_fraction(
    time_s: np.ndarray,
    values: np.ndarray,
    onset: int,
    peak: int,
    offset: int,
    baseline: float,
    amplitude: float,
    fraction_of_amplitude: float,
) -> float | None:
    """Transient width at a given fraction of the amplitude above baseline.

    fraction_of_amplitude = 0.5 -> CTD50 (width at half-amplitude);
    0.1 -> CTD90 (width at 90 % decay, i.e. the 10 % level). Rising-side
    crossing and falling-side crossing are both linearly interpolated.
    """
    if amplitude <= 0:
        return None
    level = baseline + fraction_of_amplitude * amplitude
    t_up = _rise_crossing_time(time_s, values, onset, peak, level)
    decay_after_peak = time_to_decay(time_s, values, peak, offset, baseline, amplitude, 1.0 - fraction_of_amplitude)
    if t_up is None or decay_after_peak is None:
        return None
    return float((time_s[peak] + decay_after_peak) - t_up)


def analyze_calcium(
    frames: np.ndarray,
    fps: float,
    roi_mask: np.ndarray | None = None,
    min_transients_per_min: float = 240.0,
    prominence_frac: float = 0.2,
    background_trace: np.ndarray | None = None,
    auto_background: bool = False,
    compute_map: bool = True,
) -> CalciumResult:
    """background_trace: per-frame background fluorescence (e.g. from a
    cell-free region) subtracted before F0 normalization, as in the lab's
    MATLAB pipeline; auto_background derives one from the darkest pixels
    instead (ignored if background_trace is given)."""
    raw_trace = extract_intensity_trace(frames, roi_mask)

    background_method = "none"
    bg = None
    if background_trace is not None:
        bg = np.asarray(background_trace, dtype=np.float64)
        if bg.shape != raw_trace.shape:
            raise ValueError("background_trace must have one value per frame")
        background_method = "manual_roi"
    elif auto_background:
        bg = auto_background_trace(frames)
        background_method = "auto_darkest_pixels"
    corrected = raw_trace - bg if bg is not None else raw_trace

    df_f0, _f0 = compute_df_f0(corrected)
    n = len(df_f0)
    time_s = np.arange(n) / fps

    smoothed = smooth(df_f0, fps)
    peaks = detect_peaks(
        smoothed, fps, min_bpm_gap=min_transients_per_min, prominence_frac=prominence_frac
    )

    transients = []
    for i, peak in enumerate(peaks):
        window_start = peaks[i - 1] if i > 0 else 0
        next_bound = peaks[i + 1] if i + 1 < len(peaks) else n
        onset = find_local_min_between(smoothed, window_start, peak)
        offset = find_local_min_between(smoothed, peak, next_bound)

        baseline = smoothed[onset]
        peak_val = smoothed[peak]
        amplitude = float(peak_val - baseline)

        rise_time_s = None
        if amplitude > 0:
            lo = baseline + 0.1 * amplitude
            hi = baseline + 0.9 * amplitude
            seg = smoothed[onset : peak + 1]
            seg_t = time_s[onset : peak + 1]
            above_lo = np.where(seg >= lo)[0]
            above_hi = np.where(seg >= hi)[0]
            if len(above_lo) and len(above_hi):
                rise_time_s = float(seg_t[above_hi[0]] - seg_t[above_lo[0]])

        tau_s = None
        if offset > peak + 3:
            seg_t = time_s[peak:offset] - time_s[peak]
            seg_y = smoothed[peak:offset]
            try:
                c0 = smoothed[offset]
                popt, _ = curve_fit(
                    _exp_decay,
                    seg_t,
                    seg_y,
                    p0=[max(amplitude, 1e-6), max(seg_t[-1] / 2, 1e-3), c0],
                    maxfev=2000,
                )
                if popt[1] > 0:
                    tau_s = float(popt[1])
            except Exception:
                tau_s = None

        ipi_s = float(time_s[peaks[i]] - time_s[peaks[i - 1]]) if i > 0 else None

        # Time-to-decay 10/50/90% (same definition as the beating analysis
        # and PIV-MyoMonitor): time after the peak to drop that fraction of
        # the way back to the pre-transient baseline. Complements the
        # exponential tau, which assumes a single-exponential shape.
        decay_10_s = time_to_decay(time_s, smoothed, peak, offset, baseline, amplitude, 0.10)
        decay_50_s = time_to_decay(time_s, smoothed, peak, offset, baseline, amplitude, 0.50)
        decay_90_s = time_to_decay(time_s, smoothed, peak, offset, baseline, amplitude, 0.90)

        # Release / reuptake / duration phases as reported in calcium-
        # transient figures (start-to-peak, peak-to-end, transient duration)
        # and the BeatProfiler-style width measures CTD50 / CTD90.
        start_to_peak_s = float(time_s[peak] - time_s[onset])
        peak_to_end_s = float(time_s[offset] - time_s[peak]) if offset > peak else None
        duration_s = float(time_s[offset] - time_s[onset]) if offset > onset else None
        ctd50_s = _width_at_fraction(time_s, smoothed, onset, peak, offset, baseline, amplitude, 0.5)
        ctd90_s = _width_at_fraction(time_s, smoothed, onset, peak, offset, baseline, amplitude, 0.1)

        transients.append(
            {
                "transient_index": i,
                "onset_time_s": float(time_s[onset]),
                "peak_time_s": float(time_s[peak]),
                "end_time_s": float(time_s[offset]),
                "amplitude_df_f0": amplitude,
                "start_to_peak_s": start_to_peak_s,
                "peak_to_end_s": peak_to_end_s,
                "duration_s": duration_s,
                "ctd50_s": ctd50_s,
                "ctd90_s": ctd90_s,
                "rise_time_10_90_s": rise_time_s,
                "decay_tau_s": tau_s,
                "time_to_decay_10_s": decay_10_s,
                "time_to_decay_50_s": decay_50_s,
                "time_to_decay_90_s": decay_90_s,
                "inter_peak_interval_s": ipi_s,
            }
        )

    transients_df = pd.DataFrame(transients)
    ipis = (
        transients_df["inter_peak_interval_s"].dropna().to_numpy() if len(transients_df) else np.array([])
    )
    amplitudes = transients_df["amplitude_df_f0"].to_numpy() if len(transients_df) else np.array([])
    taus = transients_df["decay_tau_s"].dropna().to_numpy() if len(transients_df) else np.array([])
    has_rows = len(transients_df) > 0

    def col_mean(col: str) -> float | None:
        return safe_mean(transients_df[col]) if has_rows else None

    summary = {
        "n_transients": int(len(peaks)),
        "duration_s": float(n / fps),
        "background_method": background_method,
        "mean_frequency_per_min": float(60.0 / ipis.mean()) if len(ipis) else None,
        "mean_frequency_hz": float(1.0 / ipis.mean()) if len(ipis) else None,
        "mean_inter_peak_interval_s": float(ipis.mean()) if len(ipis) else None,
        "mean_amplitude_df_f0": float(amplitudes.mean()) if len(amplitudes) else None,
        # "ΔF/F0 max" as reported in calcium-transient figures: the largest
        # single-transient amplitude in the recording.
        "max_amplitude_df_f0": float(amplitudes.max()) if len(amplitudes) else None,
        "amplitude_cv_percent": (
            float(100.0 * amplitudes.std() / amplitudes.mean())
            if len(amplitudes) and amplitudes.mean()
            else None
        ),
        "mean_start_to_peak_s": col_mean("start_to_peak_s"),
        "mean_peak_to_end_s": col_mean("peak_to_end_s"),
        "mean_transient_duration_s": col_mean("duration_s"),
        "mean_ctd50_s": col_mean("ctd50_s"),
        "mean_ctd90_s": col_mean("ctd90_s"),
        "mean_rise_time_10_90_s": col_mean("rise_time_10_90_s"),
        "mean_decay_tau_s": float(taus.mean()) if len(taus) else None,
        "mean_time_to_decay_10_s": col_mean("time_to_decay_10_s"),
        "mean_time_to_decay_50_s": col_mean("time_to_decay_50_s"),
        "mean_time_to_decay_90_s": col_mean("time_to_decay_90_s"),
    }

    df_f0_map = compute_pixel_df_f0_map(frames) if compute_map else None

    return CalciumResult(
        fps=fps,
        n_frames=n,
        time_s=time_s,
        raw_trace=raw_trace,
        df_f0=df_f0,
        peak_indices=peaks,
        transients_df=transients_df,
        summary=summary,
        background_trace=bg,
        df_f0_map=df_f0_map,
    )
