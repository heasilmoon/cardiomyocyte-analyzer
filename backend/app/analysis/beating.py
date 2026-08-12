"""Contraction/beating analysis for bright-field cardiomyocyte video.

Implements the same core idea as Fiji's MUSCLEMOTION plugin (Sala et al.,
2018, Circulation Research): pixel-intensity differences over time are used
as a proxy for contractile motion, and beats are detected as peaks in that
signal. Three signal modes are supported:

- "reference" (default): each frame is compared to a single resting/diastolic
  reference frame. This produces a displacement-like signal that rises during
  contraction and falls during relaxation, giving one clean peak per beat.
  This is MUSCLEMOTION's primary/recommended mode.
- "consecutive": each frame is compared to the previous frame. This produces
  a velocity-like signal that is more sensitive to fast motion but tends to
  show two peaks per beat (one for the contraction stroke, one for the
  relaxation stroke) when contraction and relaxation happen at comparable
  speed. Kept as an option for comparison/debugging.
- "piv": mean displacement-vector magnitude between consecutive frames from
  windowed cross-correlation PIV (see piv.py) — the same algorithm family as
  PIVlab's piv_FFTmulti, used by PIV-MyoMonitor for cardiac organoid
  contractility. Richer (a full 2D motion field, not just a scalar) but
  slower than the other two modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from app.analysis.piv import (
    assess_texture,
    compute_piv_field,
    compute_piv_motion_signal,
    compute_window_texture_mask,
)
from app.analysis.signal_common import (
    detect_peaks,
    estimate_dominant_period_s,
    find_local_min_between,
    smooth,
)

SignalMode = Literal["reference", "consecutive", "piv"]


@dataclass
class BeatingResult:
    fps: float
    n_frames: int
    signal_mode: str
    reference_frame_index: int | None
    time_s: np.ndarray
    raw_signal: np.ndarray
    smoothed_signal: np.ndarray
    peak_indices: np.ndarray
    trough_indices: np.ndarray
    beats_df: pd.DataFrame
    summary: dict = field(default_factory=dict)
    piv_field: dict | None = None


def _consecutive_diff_signal(frames: np.ndarray) -> np.ndarray:
    diffs = np.abs(np.diff(frames.astype(np.float32), axis=0))
    return diffs.mean(axis=(1, 2))


def _time_to_decay(
    time_s: np.ndarray,
    values: np.ndarray,
    peak_idx: int,
    end_idx: int,
    baseline: float,
    amplitude: float,
    fraction: float,
) -> float | None:
    """Time after a beat's peak for the signal to decay by `fraction` of its amplitude.

    Matches the "time-to-decay T10/T50/T90" parameters reported by
    PIV-MyoMonitor (Lee et al., 2024, Front. Bioeng. Biotechnol.) — T10 is
    the (short) time to drop 10% of the way back to baseline, T90 the
    (longer) time to drop 90% of the way. The crossing point is linearly
    interpolated between the two bracketing samples rather than snapped to
    the nearest frame, since frame spacing alone can be coarse relative to
    the decay itself at low fps.

    Returns None if the amplitude is non-positive or the signal never
    reaches the target level before `end_idx` (e.g. a truncated final beat).
    """
    if amplitude <= 0 or end_idx <= peak_idx:
        return None
    target = baseline + (1.0 - fraction) * amplitude
    for k in range(peak_idx, end_idx):
        v0, v1 = values[k], values[k + 1]
        if v1 <= target:
            if v0 == v1:
                t_cross = time_s[k]
            else:
                frac = (v0 - target) / (v0 - v1)
                t_cross = time_s[k] + frac * (time_s[k + 1] - time_s[k])
            return float(t_cross - time_s[peak_idx])
    return None


def _safe_mean(values) -> float | None:
    """Mean of a column/array, treating NaN/None as missing rather than as 0.

    Plain `.mean()` (or `.dropna().mean()` on an all-null column) returns
    NaN, not None, when every value is missing — e.g. a beat whose
    relaxation segment never actually decays 90% of the way back to
    baseline before the next beat starts. A NaN silently breaks JSON
    serialization of the response (FastAPI/Starlette reject it), so this
    treats "no valid values" as None instead of propagating NaN.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else None


def pick_reference_frame(frames: np.ndarray, fps: float = 30.0) -> int:
    """Pick a resting/diastolic frame to use as the MUSCLEMOTION-style reference.

    Finds the widest low-motion stretch of the consecutive frame-to-frame
    difference signal (a proxy for diastole, which usually lasts longer than
    the contraction stroke) and returns a frame in the middle of it. The
    signal is smoothed first so a single quiet instant — e.g. the momentary
    zero-velocity turning point at the peak of a contraction — isn't mistaken
    for the resting period, which would otherwise happen with a raw argmin.

    The first/last few percent of frames are excluded from the search: the
    Savitzky-Golay smoothing used here can undershoot right at the array
    boundary (it has less data to fit a polynomial to there), which can bias
    a plain argmin toward the very first or last frame even when that isn't
    actually the calmest part of the recording.
    """
    n = frames.shape[0]
    if n < 10:
        return 0
    consecutive = _consecutive_diff_signal(frames)
    smoothed = smooth(consecutive, fps, window_seconds=0.3)
    margin = max(int(round(0.03 * len(smoothed))), 2)
    if len(smoothed) <= 2 * margin:
        return int(np.argmin(smoothed))
    return margin + int(np.argmin(smoothed[margin:-margin]))


def compute_motion_signal(
    frames: np.ndarray,
    mode: SignalMode = "reference",
    reference_index: int | None = None,
    fps: float = 30.0,
    piv_window_size: int = 32,
    piv_step: int | None = None,
) -> tuple[np.ndarray, int | None]:
    """Motion signal used as the contraction proxy.

    frames: (N, H, W) grayscale array.
    Returns (signal, reference_frame_index). reference_frame_index is None
    for "consecutive" and "piv" modes. In "reference" mode the signal has
    length N (one value per frame, index 0 included); in "consecutive" and
    "piv" modes it has length N-1 (one value per frame-to-frame transition).
    """
    if frames.ndim != 3:
        raise ValueError("Expected grayscale frames with shape (N, H, W)")

    if mode == "consecutive":
        return _consecutive_diff_signal(frames), None

    if mode == "piv":
        texture_mask = compute_window_texture_mask(frames[0], window_size=piv_window_size, step=piv_step)
        signal = compute_piv_motion_signal(
            frames, window_size=piv_window_size, step=piv_step, texture_mask=texture_mask
        )
        return signal, None

    if mode != "reference":
        raise ValueError(f"Unknown signal mode: {mode}")

    if reference_index is None:
        reference_index = pick_reference_frame(frames, fps=fps)
    reference_index = int(np.clip(reference_index, 0, frames.shape[0] - 1))
    reference = frames[reference_index].astype(np.float32)
    signal = np.abs(frames.astype(np.float32) - reference[None, ...]).mean(axis=(1, 2))
    return signal, reference_index


def analyze_beating(
    frames: np.ndarray,
    fps: float,
    min_bpm_gap: float | None = None,
    prominence_frac: float = 0.15,
    signal_mode: SignalMode = "reference",
    reference_index: int | None = None,
    piv_window_size: int = 32,
    piv_step: int | None = None,
) -> BeatingResult:
    raw_signal, used_reference_index = compute_motion_signal(
        frames,
        mode=signal_mode,
        reference_index=reference_index,
        fps=fps,
        piv_window_size=piv_window_size,
        piv_step=piv_step,
    )
    n = len(raw_signal)
    time_s = np.arange(n) / fps

    # Auto-tune the smoothing window and minimum peak spacing to the video's
    # own dominant beat period instead of assuming a fixed, fast default.
    # A short fixed window (e.g. 0.15s) undersmooths slow hiPSC-CM beating
    # (~0.5-1.5 Hz, i.e. 0.7-2s period) enough that frame-level noise/
    # compression artifacts can pass the prominence test and get counted as
    # extra beats. Autocorrelation on the raw signal is a robust way to find
    # the true repeat period even with a smaller high-frequency contaminant
    # riding on top of it.
    estimated_period_s = estimate_dominant_period_s(raw_signal, fps)
    smoothing_window_s = float(np.clip(estimated_period_s / 6.0, 0.05, 0.5))
    if min_bpm_gap is None:
        min_bpm_gap = float(np.clip(100.0 / estimated_period_s, 30.0, 400.0))

    smoothed = smooth(raw_signal, fps, window_seconds=smoothing_window_s)
    peaks = detect_peaks(smoothed, fps, min_bpm_gap=min_bpm_gap, prominence_frac=prominence_frac)

    troughs = []
    for i, peak in enumerate(peaks):
        window_start = peaks[i - 1] if i > 0 else 0
        troughs.append(find_local_min_between(smoothed, window_start, peak))
    troughs = np.array(troughs, dtype=int)

    beats = []
    for i, peak in enumerate(peaks):
        trough_before = troughs[i]
        next_bound = peaks[i + 1] if i + 1 < len(peaks) else n
        trough_after = find_local_min_between(smoothed, peak, next_bound)

        baseline = (smoothed[trough_before] + smoothed[trough_after]) / 2.0
        amplitude = float(smoothed[peak] - baseline)
        contraction_time_s = float(time_s[peak] - time_s[trough_before])
        relaxation_time_s = float(time_s[trough_after] - time_s[peak]) if trough_after < n else None
        ibi_s = float(time_s[peaks[i]] - time_s[peaks[i - 1]]) if i > 0 else None

        contraction_segment = smoothed[trough_before : peak + 1]
        max_contraction_velocity = (
            float(np.max(np.diff(contraction_segment)) * fps) if len(contraction_segment) > 1 else None
        )
        relaxation_segment = smoothed[peak : trough_after + 1]
        max_relaxation_velocity = (
            float(-np.min(np.diff(relaxation_segment)) * fps) if len(relaxation_segment) > 1 else None
        )

        time_to_decay_10_s = _time_to_decay(time_s, smoothed, peak, trough_after, baseline, amplitude, 0.10)
        time_to_decay_50_s = _time_to_decay(time_s, smoothed, peak, trough_after, baseline, amplitude, 0.50)
        time_to_decay_90_s = _time_to_decay(time_s, smoothed, peak, trough_after, baseline, amplitude, 0.90)

        beats.append(
            {
                "beat_index": i,
                "contraction_start_time_s": float(time_s[trough_before]),
                "peak_time_s": float(time_s[peak]),
                "relaxation_end_time_s": float(time_s[trough_after]),
                "amplitude": amplitude,
                "contraction_time_s": contraction_time_s,
                "relaxation_time_s": relaxation_time_s,
                "max_contraction_velocity": max_contraction_velocity,
                "max_relaxation_velocity": max_relaxation_velocity,
                "time_to_decay_10_s": time_to_decay_10_s,
                "time_to_decay_50_s": time_to_decay_50_s,
                "time_to_decay_90_s": time_to_decay_90_s,
                "inter_beat_interval_s": ibi_s,
                "instantaneous_bpm": (60.0 / ibi_s) if ibi_s else None,
            }
        )

    beats_df = pd.DataFrame(beats)

    ibis = beats_df["inter_beat_interval_s"].dropna().to_numpy() if len(beats_df) else np.array([])
    amplitudes = beats_df["amplitude"].to_numpy() if len(beats_df) else np.array([])

    summary = {
        "signal_mode": signal_mode,
        "reference_frame_index": used_reference_index,
        "estimated_period_s": estimated_period_s,
        "smoothing_window_s": smoothing_window_s,
        "min_bpm_gap_used": min_bpm_gap,
        "n_beats": int(len(peaks)),
        "duration_s": float(n / fps),
        "mean_bpm": float(60.0 / ibis.mean()) if len(ibis) else None,
        "mean_inter_beat_interval_s": float(ibis.mean()) if len(ibis) else None,
        "ibi_std_s": float(ibis.std()) if len(ibis) else None,
        "ibi_cv_percent": float(100.0 * ibis.std() / ibis.mean()) if len(ibis) and ibis.mean() else None,
        "mean_amplitude": _safe_mean(amplitudes),
        "amplitude_cv_percent": (
            float(100.0 * amplitudes.std() / amplitudes.mean())
            if len(amplitudes) and amplitudes.mean()
            else None
        ),
        "mean_contraction_time_s": _safe_mean(beats_df["contraction_time_s"]) if len(beats_df) else None,
        "mean_relaxation_time_s": _safe_mean(beats_df["relaxation_time_s"]) if len(beats_df) else None,
        "mean_max_contraction_velocity": (
            _safe_mean(beats_df["max_contraction_velocity"]) if len(beats_df) else None
        ),
        "mean_max_relaxation_velocity": (
            _safe_mean(beats_df["max_relaxation_velocity"]) if len(beats_df) else None
        ),
        "mean_time_to_decay_10_s": _safe_mean(beats_df["time_to_decay_10_s"]) if len(beats_df) else None,
        "mean_time_to_decay_50_s": _safe_mean(beats_df["time_to_decay_50_s"]) if len(beats_df) else None,
        "mean_time_to_decay_90_s": _safe_mean(beats_df["time_to_decay_90_s"]) if len(beats_df) else None,
    }

    piv_field = None
    if signal_mode == "piv":
        texture = assess_texture(frames[0], window_size=piv_window_size, step=piv_step)
        summary["piv_median_window_std"] = texture["median_window_std"]
        summary["piv_low_texture_warning"] = texture["low_texture"]
        texture_mask = compute_window_texture_mask(frames[0], window_size=piv_window_size, step=piv_step)
        summary["piv_n_windows_total"] = int(texture_mask.size)
        summary["piv_n_low_texture_windows_masked"] = int((~texture_mask).sum())

    if signal_mode == "piv" and len(peaks) > 0:
        # Representative vector field at the strongest beat, for the
        # PIVlab-style arrow/heatmap visualization — computing and storing
        # a field for every frame pair in the video would be expensive and
        # is rarely useful beyond one illustrative example.
        strongest_peak = int(peaks[np.argmax(smoothed[peaks])])
        strongest_peak = min(strongest_peak, frames.shape[0] - 2)
        piv_field = compute_piv_field(
            frames[strongest_peak], frames[strongest_peak + 1], window_size=piv_window_size, step=piv_step
        )
        piv_field["frame_index"] = strongest_peak

    return BeatingResult(
        fps=fps,
        n_frames=int(frames.shape[0]),
        signal_mode=signal_mode,
        reference_frame_index=used_reference_index,
        time_s=time_s,
        raw_signal=raw_signal,
        smoothed_signal=smoothed,
        peak_indices=peaks,
        trough_indices=troughs,
        beats_df=beats_df,
        summary=summary,
        piv_field=piv_field,
    )
