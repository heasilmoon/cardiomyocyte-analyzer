import json

import numpy as np

from app.analysis.beating import analyze_beating, compute_motion_signal
from app.analysis.optical_flow import (
    _pair_strokes,
    analyze_speed_waves,
    compute_optical_flow_speed_signal,
    flow_to_vector_grid,
)
from tests.test_beating import _make_pulsing_frames


def _two_wave_speed(fps=100.0, n_beats=5, period_s=1.0, tbc_s=0.3):
    """Synthetic speed curve: a contraction bump then a relaxation bump per
    beat, riding on a small constant baseline."""
    t = np.arange(int(n_beats * period_s * fps)) / fps
    s = np.full_like(t, 0.2)
    for k in range(n_beats):
        c = k * period_s + 0.2
        s += 3.0 * np.exp(-(((t - c) / 0.04) ** 2))
        s += 2.0 * np.exp(-(((t - (c + tbc_s)) / 0.05) ** 2))
    return t, s


def test_speed_signal_length_and_units():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=40, fps=fps, hz=1.0, size=64, textured=True)
    px = compute_optical_flow_speed_signal(frames, fps)
    um = compute_optical_flow_speed_signal(frames, fps, um_per_px=0.5)
    assert px.shape == (39,)
    # µm/s is just px/s scaled by the pixel size.
    assert np.allclose(um, px * 0.5)
    assert px.max() > 0

    signal, ref = compute_motion_signal(frames, mode="optical_flow", fps=fps)
    assert len(signal) == 39 and ref is None


def test_flow_to_vector_grid_shape():
    flow = np.zeros((64, 96, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    grid = flow_to_vector_grid(flow, step=16, scale=3.0)
    assert grid["x"].shape == grid["u"].shape == (4, 6)
    assert np.allclose(grid["u"], 6.0)
    assert np.allclose(grid["v"], 0.0)


def test_pair_strokes_pairs_nearby_tall_peaks_and_ignores_noise_bumps():
    speed = np.zeros(300)
    # Beat 1: contraction at 20 (height 10), relaxation at 50 (height 6).
    # Beat 2: relaxation is the taller stroke: contraction at 120 (4), relaxation at 150 (10).
    # Beat 3: contraction at 220 (10) with only a tiny noise bump at 260 (1).
    for i, h in [(20, 10), (50, 6), (120, 4), (150, 10), (220, 10), (260, 1)]:
        speed[i] = h
    anchors = np.array([20, 150, 220])
    strokes = np.array([20, 50, 120, 150, 220, 260])
    pairs = _pair_strokes(speed, anchors, strokes, baseline=0.0, period_frames=100)
    assert pairs == [(20, 50), (120, 150), (220, None)]


def test_analyze_speed_waves_recovers_contractionwave_parameters():
    fps = 100.0
    t, s = _two_wave_speed(fps=fps, n_beats=5, period_s=1.0, tbc_s=0.3)
    waves = analyze_speed_waves(t, s, fps, period_s=1.0, prominence_frac=0.15, threshold_frac=0.10)
    df = waves["cycles_df"]
    assert len(df) == 5
    assert df["complete_wave"].all()
    assert len(waves["relaxation_idx"]) == 5

    for _, row in df.iterrows():
        # Ordering: start < MCS < dip < MRS < end.
        assert (
            row["wave_start_time_s"]
            < row["contraction_peak_time_s"]
            < row["min_speed_time_s"]
            < row["relaxation_peak_time_s"]
            < row["wave_end_time_s"]
        )
        # ContractionWave identities: CT = CTP + CTPMS, RT = RTP + RTPB, CRT = CT + RT.
        assert abs(row["contraction_time_to_peak_s"] + row["contraction_peak_to_min_speed_s"] - row["contraction_time_s"]) < 1e-9
        assert abs(row["relaxation_time_to_peak_s"] + row["relaxation_peak_to_baseline_s"] - row["relaxation_time_s"]) < 1e-9
        assert abs(row["contraction_time_s"] + row["relaxation_time_s"] - row["contraction_relaxation_time_s"]) < 1e-9
        # Known synthetic layout.
        assert abs(row["time_between_max_speeds_s"] - 0.3) < 0.03
        assert abs(row["max_contraction_speed"] - 3.0) < 0.15
        assert abs(row["max_relaxation_speed"] - 2.0) < 0.15
        assert row["mcs_mrs_difference"] > 0
        assert 0 < row["shortening_area"] < row["contraction_relaxation_area"]

    ibi = df["inter_beat_interval_s"].dropna()
    assert np.allclose(ibi, 1.0, atol=0.02)


def test_analyze_beating_optical_flow_mode_end_to_end():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=240, fps=fps, hz=1.0, size=96, textured=True)
    result = analyze_beating(frames, fps, signal_mode="optical_flow", um_per_px=0.5)
    s = result.summary
    assert s["signal_mode"] == "optical_flow"
    assert s["speed_units"] == "um/s"
    assert s["area_units"] == "um"
    # 1 Hz pulsing for 8 s -> ~8 beats, ~60 BPM.
    assert 6 <= s["n_beats"] <= 9
    assert abs(s["mean_bpm"] - 60.0) < 6.0
    assert s["mean_max_contraction_speed"] > 0
    assert s["mean_contraction_relaxation_time_s"] > 0
    assert set(result.beats_df.columns) >= {
        "max_contraction_speed",
        "max_relaxation_speed",
        "contraction_time_to_peak_s",
        "contraction_relaxation_time_s",
        "time_between_max_speeds_s",
        "contraction_relaxation_area",
        "shortening_area",
        "inter_beat_interval_s",
    }
    assert result.secondary_peak_indices is not None
    assert result.piv_field is not None and result.piv_field["kind"] == "optical_flow"
    json.dumps(s, allow_nan=False)


def test_analyze_beating_optical_flow_handles_static_video():
    fps = 30.0
    frames = np.full((40, 32, 32), 100, dtype=np.uint8)
    result = analyze_beating(frames, fps, signal_mode="optical_flow")
    assert result.summary["n_beats"] == 0
    assert result.summary["mean_bpm"] is None
    json.dumps(result.summary, allow_nan=False)
