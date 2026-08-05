import numpy as np

from app.analysis.beating import analyze_beating, compute_motion_signal


def _make_pulsing_frames(n_frames=180, fps=30.0, hz=1.0, size=40):
    frames = np.zeros((n_frames, size, size), dtype=np.uint8)
    t = np.arange(n_frames) / fps
    radius = 10 + 5 * np.clip(np.sin(2 * np.pi * hz * t), 0, None)
    yy, xx = np.mgrid[0:size, 0:size]
    center = size // 2
    for i, r in enumerate(radius):
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= r**2
        frames[i][mask] = 200
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
