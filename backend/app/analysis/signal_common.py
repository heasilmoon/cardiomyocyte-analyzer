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


def estimate_dominant_period_s(
    signal: np.ndarray,
    fps: float,
    min_bpm: float = 15.0,
    max_bpm: float = 200.0,
) -> float:
    """Estimate the signal's dominant repeat period via autocorrelation.

    Restricting the search to a physiologically plausible band (default
    15-200 BPM, well above typical hiPSC-CM range but well below e.g. mouse
    heart rate) keeps this from locking onto sub-beat noise or single-frame
    artifacts, and it's fairly robust to a small high-frequency contaminant
    riding on top of the real beat waveform since autocorrelation averages
    over the whole signal rather than looking at any single point.

    Returns the period in seconds; falls back to 1.0s (60 BPM) if the signal
    is too short or has no variation to measure.
    """
    n = len(signal)
    if n < 10:
        return 1.0
    x = signal.astype(np.float64) - np.mean(signal)
    if np.allclose(x, 0):
        return 1.0
    autocorr = np.correlate(x, x, mode="full")[n - 1 :]
    if autocorr[0] <= 0:
        return 1.0
    autocorr = autocorr / autocorr[0]

    min_lag = max(int(round(fps * 60.0 / max_bpm)), 1)
    max_lag = min(int(round(fps * 60.0 / min_bpm)), n - 1)
    if max_lag <= min_lag:
        return 1.0

    segment = autocorr[min_lag : max_lag + 1]
    best_lag = min_lag + int(np.argmax(segment))
    return best_lag / fps


def find_local_min_between(signal: np.ndarray, start: int, end: int) -> int:
    """Index of the minimum value of signal within [start, end)."""
    start = max(start, 0)
    end = min(end, len(signal))
    if end <= start:
        return start
    segment = signal[start:end]
    return start + int(np.argmin(segment))
