import numpy as np

from app.analysis.calcium import analyze_calcium, compute_df_f0, extract_intensity_trace


def _make_transient_frames(n_frames=180, fps=30.0, hz=1.0, size=20):
    t = np.arange(n_frames) / fps
    phase = (t * hz) % 1.0
    intensity = 60 + 150 * np.exp(-((phase * 6) ** 2))
    frames = np.tile(intensity[:, None, None], (1, size, size)).astype(np.uint8)
    return frames


def test_extract_intensity_trace_shape():
    frames = _make_transient_frames(n_frames=10)
    trace = extract_intensity_trace(frames)
    assert trace.shape == (10,)


def test_compute_df_f0_baseline_near_zero():
    trace = np.array([10.0] * 10 + [100.0])
    df_f0, f0 = compute_df_f0(trace, baseline_percentile=10.0)
    assert f0 == 10.0
    assert df_f0[-1] > 0


def test_analyze_calcium_detects_transients():
    fps = 30.0
    frames = _make_transient_frames(n_frames=180, fps=fps, hz=1.0)
    result = analyze_calcium(frames, fps)
    assert result.summary["n_transients"] > 0
    assert set(result.transients_df.columns) >= {
        "peak_time_s",
        "amplitude_df_f0",
        "rise_time_10_90_s",
        "decay_tau_s",
        "time_to_decay_10_s",
        "time_to_decay_50_s",
        "time_to_decay_90_s",
    }


def test_analyze_calcium_reports_decay_times_max_amplitude_and_hz():
    import json

    fps = 30.0
    frames = _make_transient_frames(n_frames=180, fps=fps, hz=1.0)
    result = analyze_calcium(frames, fps)
    s = result.summary

    # 1 Hz synthetic transients -> ~1 Hz / ~60 per min, and the two units agree.
    assert abs(s["mean_frequency_hz"] - 1.0) < 0.1
    assert abs(s["mean_frequency_per_min"] / 60.0 - s["mean_frequency_hz"]) < 1e-9

    # Max is the largest single-transient amplitude, never below the mean.
    assert s["max_amplitude_df_f0"] >= s["mean_amplitude_df_f0"]
    assert s["max_amplitude_df_f0"] == result.transients_df["amplitude_df_f0"].max()

    # Deeper decay takes longer: T10 <= T50 <= T90 wherever all three exist.
    for _, tr in result.transients_df.iterrows():
        t10, t50, t90 = tr["time_to_decay_10_s"], tr["time_to_decay_50_s"], tr["time_to_decay_90_s"]
        if t10 is not None and t50 is not None:
            assert t10 <= t50
        if t50 is not None and t90 is not None:
            assert t50 <= t90
    assert s["mean_time_to_decay_10_s"] is not None
    assert s["mean_time_to_decay_50_s"] is not None

    json.dumps(s, allow_nan=False)  # no NaN leaking into the API response


def test_analyze_calcium_phase_durations_and_widths_are_consistent():
    fps = 30.0
    frames = _make_transient_frames(n_frames=180, fps=fps, hz=1.0)
    result = analyze_calcium(frames, fps)
    df = result.transients_df
    assert set(df.columns) >= {
        "onset_time_s", "end_time_s", "start_to_peak_s", "peak_to_end_s", "duration_s", "ctd50_s", "ctd90_s"
    }
    for _, tr in df.iterrows():
        assert tr["onset_time_s"] <= tr["peak_time_s"] <= tr["end_time_s"]
        if tr["peak_to_end_s"] is not None and tr["duration_s"] is not None:
            assert abs(tr["start_to_peak_s"] + tr["peak_to_end_s"] - tr["duration_s"]) < 1e-9
        # Width at half-amplitude is narrower than width at 90 % decay, and
        # both fit inside the whole transient.
        if tr["ctd50_s"] is not None and tr["ctd90_s"] is not None:
            assert tr["ctd50_s"] <= tr["ctd90_s"] <= tr["duration_s"] + 1e-9
    assert result.summary["mean_ctd50_s"] is not None
    assert result.summary["mean_start_to_peak_s"] > 0


def test_analyze_calcium_background_subtraction_and_map():
    fps = 30.0
    frames = _make_transient_frames(n_frames=120, fps=fps, hz=1.0)
    # Add a constant dark border (cell-free background) around the signal.
    padded = np.zeros((frames.shape[0], 40, 40), dtype=np.uint8) + 20
    padded[:, 10:30, 10:30] = frames
    bg_trace = np.full(frames.shape[0], 20.0)

    manual = analyze_calcium(padded, fps, background_trace=bg_trace)
    assert manual.summary["background_method"] == "manual_roi"
    assert manual.background_trace is not None
    auto = analyze_calcium(padded, fps, auto_background=True)
    assert auto.summary["background_method"] == "auto_darkest_pixels"
    # The darkest 5 % of pixels are the constant-20 border -> ~20 everywhere.
    assert np.allclose(auto.background_trace, 20.0, atol=1.0)
    none = analyze_calcium(padded, fps)
    assert none.summary["background_method"] == "none"
    assert none.background_trace is None

    # Pixel map has frame shape and is largest inside the signal region.
    assert auto.df_f0_map.shape == (40, 40)
    assert auto.df_f0_map[15:25, 15:25].mean() > auto.df_f0_map[:5, :5].mean()

    try:
        analyze_calcium(padded, fps, background_trace=np.zeros(5))
        assert False, "expected ValueError for wrong-length background trace"
    except ValueError:
        pass
