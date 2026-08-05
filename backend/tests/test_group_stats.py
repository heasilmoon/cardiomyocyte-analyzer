from app.analysis.group_stats import compare_groups


def _summary(bpm, amplitude, extra=None):
    d = {"mean_bpm": bpm, "mean_amplitude": amplitude, "n_beats": 6}
    if extra:
        d.update(extra)
    return d


def test_compare_groups_detects_real_difference():
    group_a = [_summary(bpm, 1.0) for bpm in [58, 60, 61, 59, 60, 62]]
    group_b = [_summary(bpm, 1.0) for bpm in [88, 91, 90, 89, 92, 90]]

    result = compare_groups(group_a, group_b, "Control", "Treatment")

    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert bpm_row["mean_a"] < bpm_row["mean_b"]
    assert bpm_row["p_value"] is not None
    assert bpm_row["p_value"] < 0.05


def test_compare_groups_no_difference_stays_nonsignificant():
    group_a = [_summary(60, 1.0), _summary(61, 1.05), _summary(59, 0.95), _summary(60, 1.0)]
    group_b = [_summary(60, 1.0), _summary(59, 0.98), _summary(61, 1.02), _summary(60, 1.0)]

    result = compare_groups(group_a, group_b)
    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert bpm_row["p_value"] is None or bpm_row["p_value"] > 0.05


def test_compare_groups_excludes_bookkeeping_fields():
    group_a = [_summary(60, 1.0, {"reference_frame_index": 12})]
    group_b = [_summary(90, 1.0, {"reference_frame_index": 400})]

    result = compare_groups(group_a, group_b)
    metric_names = {m["metric"] for m in result["metrics"]}
    assert "reference_frame_index" not in metric_names
    assert "mean_bpm" in metric_names


def test_compare_groups_handles_missing_metric_gracefully():
    group_a = [{"mean_bpm": 60}, {"mean_bpm": 62}]
    group_b = [{"mean_bpm": 90}]  # only one group has this metric fully present

    result = compare_groups(group_a, group_b)
    assert result["n_videos_a"] == 2
    assert result["n_videos_b"] == 1
    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert bpm_row["n_a"] == 2
    assert bpm_row["n_b"] == 1
