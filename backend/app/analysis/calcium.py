"""Calcium-imaging transient analysis.

Extracts a mean-intensity trace (whole frame, or a supplied ROI mask),
converts it to a normalized dF/F0 trace, then detects individual calcium
transients and characterizes each one (amplitude, time-to-peak, rise time,
decay time constant).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from app.analysis.signal_common import detect_peaks, find_local_min_between, smooth


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


def compute_df_f0(trace: np.ndarray, baseline_percentile: float = 10.0) -> tuple[np.ndarray, float]:
    f0 = float(np.percentile(trace, baseline_percentile))
    f0 = f0 if f0 > 1e-6 else 1e-6
    return (trace - f0) / f0, f0


def analyze_calcium(
    frames: np.ndarray,
    fps: float,
    roi_mask: np.ndarray | None = None,
    min_transients_per_min: float = 240.0,
    prominence_frac: float = 0.2,
) -> CalciumResult:
    raw_trace = extract_intensity_trace(frames, roi_mask)
    df_f0, _f0 = compute_df_f0(raw_trace)
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

        transients.append(
            {
                "transient_index": i,
                "peak_time_s": float(time_s[peak]),
                "amplitude_df_f0": amplitude,
                "rise_time_10_90_s": rise_time_s,
                "decay_tau_s": tau_s,
                "inter_peak_interval_s": ipi_s,
            }
        )

    transients_df = pd.DataFrame(transients)
    ipis = (
        transients_df["inter_peak_interval_s"].dropna().to_numpy() if len(transients_df) else np.array([])
    )
    amplitudes = transients_df["amplitude_df_f0"].to_numpy() if len(transients_df) else np.array([])
    taus = transients_df["decay_tau_s"].dropna().to_numpy() if len(transients_df) else np.array([])

    summary = {
        "n_transients": int(len(peaks)),
        "duration_s": float(n / fps),
        "mean_frequency_per_min": float(60.0 / ipis.mean()) if len(ipis) else None,
        "mean_inter_peak_interval_s": float(ipis.mean()) if len(ipis) else None,
        "mean_amplitude_df_f0": float(amplitudes.mean()) if len(amplitudes) else None,
        "amplitude_cv_percent": (
            float(100.0 * amplitudes.std() / amplitudes.mean())
            if len(amplitudes) and amplitudes.mean()
            else None
        ),
        "mean_decay_tau_s": float(taus.mean()) if len(taus) else None,
    }

    return CalciumResult(
        fps=fps,
        n_frames=n,
        time_s=time_s,
        raw_trace=raw_trace,
        df_f0=df_f0,
        peak_indices=peaks,
        transients_df=transients_df,
        summary=summary,
    )
