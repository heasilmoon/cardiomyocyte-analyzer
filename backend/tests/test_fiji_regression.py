"""Regression test against a real Fiji MUSCLEMOTION output curve.

tests/fixtures/fiji_musclemotion_reference_signal.tsv is an actual
contraction-amplitude trace exported from Fiji's MUSCLEMOTION plugin by a
user of this app (~10s hiPSC-CM bright-field recording, 60 fps). By eye it
has 5-6 clear beats at roughly 30 BPM.

This app's own peak-detection pipeline (smoothing window and minimum peak
spacing auto-tuned to the signal's dominant period, see
app.analysis.signal_common.estimate_dominant_period_s) previously locked
onto frame-level noise on this kind of slow recording and reported ~240 BPM
/ 42 beats instead of ~30 BPM / ~5-6 beats — see the beating.py history for
the fix. This test feeds the exact same real trace through smooth() +
detect_peaks() (bypassing video decoding, since we only have the exported
1D curve, not the source video) to make sure that regression can't come
back silently.
"""
from pathlib import Path

import numpy as np

from app.analysis.signal_common import detect_peaks, estimate_dominant_period_s, smooth

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fiji_musclemotion_reference_signal.tsv"


def _load_fixture():
    times_ms, values = [], []
    for line in FIXTURE_PATH.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        times_ms.append(float(parts[0]))
        values.append(float(parts[1]))
    times_ms = np.array(times_ms)
    signal = np.array(values)
    fps = 1000.0 / np.median(np.diff(times_ms))
    return signal, fps


def test_fiji_signal_period_estimate_matches_visual_beat_rate():
    signal, fps = _load_fixture()
    period_s = estimate_dominant_period_s(signal, fps)
    bpm = 60.0 / period_s
    # Visually ~5-6 beats over ~10s -> roughly 28-36 BPM.
    assert 25 <= bpm <= 40


def test_fiji_signal_peak_count_matches_visual_count():
    signal, fps = _load_fixture()
    period_s = estimate_dominant_period_s(signal, fps)
    window_s = float(np.clip(period_s / 6.0, 0.05, 0.5))
    min_bpm_gap = float(np.clip(100.0 / period_s, 30.0, 400.0))

    smoothed = smooth(signal, fps, window_seconds=window_s)
    peaks = detect_peaks(smoothed, fps, min_bpm_gap=min_bpm_gap, prominence_frac=0.15)

    # Regression guard: this used to come out to ~42 spurious peaks (~240
    # BPM) on this exact recording before the adaptive-period fix.
    assert 4 <= len(peaks) <= 7
