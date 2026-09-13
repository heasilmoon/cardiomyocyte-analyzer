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
