"""Shared 1D signal helpers used by the beating and calcium modules."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def smooth(signal: np.ndarray, fps: float, window_seconds: float = 0.15) -> np.ndarray:
    """Savitzky-Golay smoothing with a window sized relative to the frame rate.

    Falls back to the raw signal when there aren't enough samples for a
    stable polynomial fit (very short clips).
    """
    window = int(round(window_seconds * fps))
    window = max(window, 5)
    if window % 2 == 0:
        window += 1
    if window >= len(signal):
        window = len(signal) - 1 if len(signal) % 2 == 0 else len(signal)
        window = max(window, 3)
    if window < 5 or window >= len(signal):
        return signal.astype(float)
    polyorder = min(3, window - 1)
    return savgol_filter(signal, window_length=window, polyorder=polyorder)


def detect_peaks(
    signal: np.ndarray,
    fps: float,
    min_bpm_gap: float = 300.0,
    prominence_frac: float = 0.15,
) -> np.ndarray:
    """Find beat/transient peaks.

    min_bpm_gap: the fastest plausible rate (beats or transients per minute);
    used to set a minimum distance between accepted peaks so noise doesn't
    get double-counted as separate beats.
    prominence_frac: required peak prominence, as a fraction of the signal's
    dynamic range, to reject low-amplitude noise.
    """
    if len(signal) < 3:
        return np.array([], dtype=int)
    min_distance = max(int(round((60.0 / min_bpm_gap) * fps)), 1)
    dynamic_range = float(np.max(signal) - np.min(signal))
    prominence = max(dynamic_range * prominence_frac, 1e-9)
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
    return peaks


def find_local_min_between(signal: np.ndarray, start: int, end: int) -> int:
    """Index of the minimum value of signal within [start, end)."""
    start = max(start, 0)
    end = min(end, len(signal))
    if end <= start:
        return start
    segment = signal[start:end]
    return start + int(np.argmin(segment))
