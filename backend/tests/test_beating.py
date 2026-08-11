import numpy as np

from app.analysis.beating import analyze_beating, compute_motion_signal


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
