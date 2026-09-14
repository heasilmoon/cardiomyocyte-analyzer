import numpy as np

from app.analysis.signal_common import (
    detect_peaks,
    estimate_dominant_period_s,
    find_local_min_between,
    smooth,
)


def _pulse_train(beat_times, fps=30.0, duration_s=10.0):
    """Narrow contraction-like pulses (fast rise, slower decay) at given times."""
    t = np.arange(0, duration_s, 1 / fps)
    y = np.zeros_like(t)
    for tb in beat_times:
        y += np.where(t < tb, np.exp(-((t - tb) / 0.06) ** 2), np.exp(-(t - tb) / 0.15))
    return y


def test_estimate_dominant_period_regular_beats():
    y = _pulse_train(np.arange(1.0, 10.0, 2.0))  # 30 BPM, perfectly regular
    assert abs(estimate_dominant_period_s(y, 30.0) - 2.0) < 0.15


def test_estimate_dominant_period_does_not_double_on_alternating_intervals():
    # Long/short alternating intervals (2.14 / 1.89 s) make the two-beat lag
    # line up better than the one-beat lag in the autocorrelation. Without a
    # subharmonic check this returned ~4.0 s and the beat detector then
    # suppressed every other beat (found by benchmarks/beating_accuracy.py).
    beats = np.array([1.61, 3.75, 5.64, 7.78, 9.72])
    y = _pulse_train(beats)
    period = estimate_dominant_period_s(y, 30.0)
    assert 1.7 < period < 2.4, period


def test_smooth_preserves_length():
    signal = np.sin(np.linspace(0, 20, 200)) + np.random.normal(0, 0.05, 200)
    smoothed = smooth(signal, fps=30.0)
    assert len(smoothed) == len(signal)


def test_smooth_handles_short_signal():
    signal = np.array([1.0, 2.0, 3.0])
    smoothed = smooth(signal, fps=30.0)
    assert len(smoothed) == len(signal)


def test_detect_peaks_finds_expected_count():
    fps = 30.0
    t = np.arange(0, 10, 1 / fps)
    # 2 Hz sine -> 20 peaks in 10 seconds
    signal = np.sin(2 * np.pi * 2.0 * t)
    peaks = detect_peaks(signal, fps, min_bpm_gap=300.0, prominence_frac=0.3)
    assert 18 <= len(peaks) <= 20


def test_find_local_min_between():
    signal = np.array([5, 4, 1, 3, 6, 0, 2])
    idx = find_local_min_between(signal, 0, 4)
    assert idx == 2
