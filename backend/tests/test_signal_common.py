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


def test_despike_removes_periodic_keyframe_comb_but_keeps_beats():
    from app.analysis.signal_common import despike_single_frame

    fps = 60.0
    t = np.arange(int(50 * fps)) / fps
    # Two real beats (multi-frame bumps) on a small noisy baseline ...
    sig = 0.3 + 0.02 * np.random.default_rng(0).normal(size=t.size)
    for tc in (20.0, 36.0):
        sig += 1.7 * np.exp(-(((t - tc) / 0.12) ** 2))
    # ... plus a single-frame spike every 30 frames (H.264 keyframe interval).
    sig[::30] += 0.8
    cleaned, n_spikes, periodic = despike_single_frame(sig)
    assert periodic is True
    assert n_spikes >= 90  # one run per keyframe (100 in 50 s)
    # Beats survive, comb is gone.
    assert cleaned[int(20 * fps)] > 1.5 and cleaned[int(36 * fps)] > 1.5
    comb = np.arange(30, t.size - 30, 30)  # skip the boundary samples, which the median cannot judge
    comb = comb[(np.abs(t[comb] - 20.0) > 1.0) & (np.abs(t[comb] - 36.0) > 1.0)]  # skip the real beats
    assert np.max(np.abs(cleaned[comb] - 0.3)) < 0.15


def test_detrend_baseline_removes_slow_drift():
    from app.analysis.signal_common import detrend_baseline

    fps = 30.0
    t = np.arange(int(40 * fps)) / fps
    drift = 3.0 * np.sin(2 * np.pi * t / 80.0)  # very slow
    beats = np.zeros_like(t)
    for tc in np.arange(1, 40, 1.0):
        beats += np.exp(-(((t - tc) / 0.08) ** 2))
    out = detrend_baseline(drift + beats, fps, window_s=4.0)
    # Baseline is now flat near 0 and beats keep their height.
    quiet = out[(t % 1.0 > 0.4) & (t % 1.0 < 0.6)]
    # Raw quiet samples span ~6 units of drift; after detrending they sit near zero.
    assert np.ptp(quiet) < 1.0 and np.abs(np.median(quiet)) < 0.35
    assert out[int(10 * fps)] > 0.8
