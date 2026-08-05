"""Contraction/beating analysis for bright-field cardiomyocyte video.

Approach (similar in spirit to MUSCLEMOTION / pixel-differencing methods
used in the cardiomyocyte literature): sum of absolute pixel-intensity
differences between consecutive frames is used as a proxy for contractile
motion. Peaks in that signal correspond to individual beats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.analysis.signal_common import detect_peaks, find_local_min_between, smooth


@dataclass
class BeatingResult:
    fps: float
    n_frames: int
    time_s: np.ndarray
    raw_signal: np.ndarray
    smoothed_signal: np.ndarray
    peak_indices: np.ndarray
    trough_indices: np.ndarray
    beats_df: pd.DataFrame
    summary: dict = field(default_factory=dict)


def compute_motion_signal(frames: np.ndarray) -> np.ndarray:
    """Frame-to-frame absolute-difference signal, one value per transition.

    frames: (N, H, W) grayscale array.
    Returns an array of length N-1; index i corresponds to the transition
    between frame i and frame i+1.
    """
    if frames.ndim != 3:
        raise ValueError("Expected grayscale frames with shape (N, H, W)")
    diffs = np.abs(np.diff(frames.astype(np.float32), axis=0))
    signal = diffs.mean(axis=(1, 2))
    return signal


def analyze_beating(
    frames: np.ndarray,
    fps: float,
    min_bpm_gap: float = 300.0,
    prominence_frac: float = 0.15,
) -> BeatingResult:
    raw_signal = compute_motion_signal(frames)
    n = len(raw_signal)
    time_s = np.arange(n) / fps

    smoothed = smooth(raw_signal, fps)
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

        beats.append(
            {
                "beat_index": i,
                "peak_time_s": float(time_s[peak]),
                "amplitude": amplitude,
                "contraction_time_s": contraction_time_s,
                "relaxation_time_s": relaxation_time_s,
                "inter_beat_interval_s": ibi_s,
                "instantaneous_bpm": (60.0 / ibi_s) if ibi_s else None,
            }
        )

    beats_df = pd.DataFrame(beats)

    ibis = beats_df["inter_beat_interval_s"].dropna().to_numpy() if len(beats_df) else np.array([])
    amplitudes = beats_df["amplitude"].to_numpy() if len(beats_df) else np.array([])

    summary = {
        "n_beats": int(len(peaks)),
        "duration_s": float(n / fps),
        "mean_bpm": float(60.0 / ibis.mean()) if len(ibis) else None,
        "mean_inter_beat_interval_s": float(ibis.mean()) if len(ibis) else None,
        "ibi_std_s": float(ibis.std()) if len(ibis) else None,
        "ibi_cv_percent": float(100.0 * ibis.std() / ibis.mean()) if len(ibis) and ibis.mean() else None,
        "mean_amplitude": float(amplitudes.mean()) if len(amplitudes) else None,
        "amplitude_cv_percent": (
            float(100.0 * amplitudes.std() / amplitudes.mean())
            if len(amplitudes) and amplitudes.mean()
            else None
        ),
        "mean_contraction_time_s": (
            float(beats_df["contraction_time_s"].mean()) if len(beats_df) else None
        ),
        "mean_relaxation_time_s": (
            float(beats_df["relaxation_time_s"].dropna().mean()) if len(beats_df) else None
        ),
    }

    return BeatingResult(
        fps=fps,
        n_frames=int(frames.shape[0]),
        time_s=time_s,
        raw_signal=raw_signal,
        smoothed_signal=smoothed,
        peak_indices=peaks,
        trough_indices=troughs,
        beats_df=beats_df,
        summary=summary,
    )
