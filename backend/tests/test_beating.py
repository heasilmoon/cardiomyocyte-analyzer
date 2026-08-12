import json

import numpy as np

from app.analysis.beating import _safe_mean, analyze_beating, compute_motion_signal


def _make_pulsing_frames(n_frames=180, fps=30.0, hz=1.0, size=40, textured=False):
    frames = np.zeros((n_frames, size, size), dtype=np.uint8)
    t = np.arange(n_frames) / fps
    radius = 10 + 5 * np.clip(np.sin(2 * np.pi * hz * t), 0, None)
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    for i, r in enumerate(radius):
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= r**2
        frames[i][mask] = 200
    if textured:
        # PIV needs correlatable texture/speckle, not flat regions — a
        # uniform-fill disk on a flat background has no interior features
        # for the interrogation windows to lock onto.
        rng = np.random.default_rng(0)
        frames = np.clip(
            frames.astype(np.int16) + rng.integers(-15, 15, frames.shape), 0, 255
        ).astype(np.uint8)
    return frames


def test_compute_motion_signal_consecutive_length():
    frames = _make_pulsing_frames(n_frames=50)
    signal, reference_index = compute_motion_signal(frames, mode="consecutive")
    assert len(signal) == 49
    assert reference_index is None


def test_compute_motion_signal_reference_length():
    frames = _make_pulsing_frames(n_frames=50)
    signal, reference_index = compute_motion_signal(frames, mode="reference")
    assert len(signal) == 50
    assert reference_index is not None


def test_analyze_beating_detects_beats_reference_mode():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=180, fps=fps, hz=1.0)
    result = analyze_beating(frames, fps, signal_mode="reference")
    assert result.summary["n_beats"] > 0
    assert result.summary["duration_s"] > 0
    assert set(result.beats_df.columns) >= {
        "peak_time_s",
        "amplitude",
        "contraction_time_s",
        "relaxation_time_s",
        "max_contraction_velocity",
        "max_relaxation_velocity",
    }


def test_analyze_beating_reference_mode_gives_one_peak_per_cycle():
    # A single sharp contraction/relaxation cycle per second should yield
    # ~1 beat/s in reference mode, unlike consecutive mode which tends to
    # double-count (one peak for the contraction stroke, one for relaxation).
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=180, fps=fps, hz=1.0)
    result = analyze_beating(frames, fps, signal_mode="reference")
    assert 5 <= result.summary["n_beats"] <= 7


def test_analyze_beating_handles_static_video():
    frames = np.full((30, 20, 20), 128, dtype=np.uint8)
    result = analyze_beating(frames, fps=30.0)
    assert result.summary["n_beats"] == 0


def test_compute_motion_signal_piv_length():
    frames = _make_pulsing_frames(n_frames=50, size=64, textured=True)
    signal, reference_index = compute_motion_signal(frames, mode="piv", piv_window_size=16, piv_step=8)
    assert len(signal) == 49
    assert reference_index is None


def test_analyze_beating_piv_mode_detects_correct_bpm():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=90, fps=fps, hz=1.0, size=64, textured=True)
    result = analyze_beating(frames, fps, signal_mode="piv", piv_window_size=16, piv_step=8)
    assert result.summary["n_beats"] >= 2
    assert abs(result.summary["mean_bpm"] - 60.0) < 5.0


def test_analyze_beating_piv_mode_stores_representative_vector_field():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=90, fps=fps, hz=1.0, size=64, textured=True)
    result = analyze_beating(frames, fps, signal_mode="piv", piv_window_size=16, piv_step=8)
    assert result.piv_field is not None
    assert result.piv_field["u"].shape == result.piv_field["v"].shape == result.piv_field["x"].shape
    assert "frame_index" in result.piv_field


def test_analyze_beating_non_piv_modes_have_no_vector_field():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=90, fps=fps, hz=1.0)
    result = analyze_beating(frames, fps, signal_mode="reference")
    assert result.piv_field is None


def test_safe_mean_treats_all_nan_as_none_not_nan():
    assert _safe_mean([]) is None
    assert _safe_mean([float("nan"), float("nan")]) is None
    assert _safe_mean([1.0, float("nan"), 3.0]) == 2.0


def test_analyze_beating_summary_always_json_serializable_across_crops():
    # Regression test: analyzing a small/awkward crop of a video (e.g. from
    # the frontend's ROI selector) previously produced NaN in summary
    # fields like mean_time_to_decay_90_s whenever every beat's value for
    # that field was missing (a `.dropna().mean()` on an all-null column
    # returns NaN, not None) - NaN isn't valid JSON and broke the endpoint
    # with a 500. json.dumps(..., allow_nan=False) mirrors what
    # Starlette's JSONResponse does, so this raises exactly if that regresses.
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=90, fps=fps, hz=1.0, size=64, textured=True)
    crops = [
        frames,
        frames[:, 5:20, 5:20],
        frames[:, 16:42, 16:41],
        frames[:, 30:64, 30:64],
        frames[:20],
    ]
    for cropped in crops:
        for mode in ("reference", "consecutive"):
            result = analyze_beating(cropped, fps, signal_mode=mode)
            json.dumps(result.summary, allow_nan=False)


def test_analyze_beating_reports_decay_times_and_start_end_timestamps():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=180, fps=fps, hz=1.0)
    result = analyze_beating(frames, fps, signal_mode="reference")
    assert set(result.beats_df.columns) >= {
        "contraction_start_time_s",
        "relaxation_end_time_s",
        "time_to_decay_10_s",
        "time_to_decay_50_s",
        "time_to_decay_90_s",
    }
    for _, beat in result.beats_df.iterrows():
        assert beat["contraction_start_time_s"] <= beat["peak_time_s"] <= beat["relaxation_end_time_s"]
        t10, t50, t90 = beat["time_to_decay_10_s"], beat["time_to_decay_50_s"], beat["time_to_decay_90_s"]
        if t10 is not None and t50 is not None:
            assert t10 <= t50
        if t50 is not None and t90 is not None:
            assert t50 <= t90
    assert result.summary["mean_time_to_decay_10_s"] is not None
    assert result.summary["mean_time_to_decay_50_s"] is not None


def test_analyze_beating_piv_mode_warns_on_flat_untextured_video():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=90, fps=fps, hz=1.0, size=64, textured=False)
    result = analyze_beating(frames, fps, signal_mode="piv", piv_window_size=16, piv_step=8)
    assert result.summary["piv_low_texture_warning"] is True


def test_analyze_beating_piv_mode_no_warning_on_textured_video():
    fps = 30.0
    frames = _make_pulsing_frames(n_frames=90, fps=fps, hz=1.0, size=64, textured=True)
    result = analyze_beating(frames, fps, signal_mode="piv", piv_window_size=16, piv_step=8)
    assert result.summary["piv_low_texture_warning"] is False
