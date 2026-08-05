import numpy as np

from app.analysis.signal_common import detect_peaks, find_local_min_between, smooth


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
