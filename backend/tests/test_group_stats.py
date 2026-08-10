import numpy as np

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


def test_lmm_fields_absent_without_cluster_labels():
    group_a = [_summary(bpm, 1.0) for bpm in [58, 60, 61, 59, 60, 62]]
    group_b = [_summary(bpm, 1.0) for bpm in [88, 91, 90, 89, 92, 90]]

    result = compare_groups(group_a, group_b)
    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert "lmm_p_value" not in bpm_row


def _clustered_dataset(seed, n_clusters=8, n_per_cluster=20, cluster_sd=4.0, noise_sd=1.0):
    """Videos from n_clusters biological samples/batches, split randomly
    into two groups with *no* true group-level effect in the generating
    model — only cluster-to-cluster variation. Returns (summaries_a,
    clusters_a, summaries_b, clusters_b)."""
    rng = np.random.default_rng(seed)
    cluster_effects = rng.normal(0, cluster_sd, n_clusters)
    labels = [f"C{i}" for i in range(n_clusters)]
    perm = rng.permutation(n_clusters)
    half = n_clusters // 2
    a_idx, b_idx = perm[:half], perm[half:]

    def build(indices):
        summaries, clusters = [], []
        for i in indices:
            for _ in range(n_per_cluster):
                val = 50 + cluster_effects[i] + rng.normal(0, noise_sd)
                summaries.append({"metric_x": val})
                clusters.append(labels[i])
        return summaries, clusters

    summaries_a, clusters_a = build(a_idx)
    summaries_b, clusters_b = build(b_idx)
    return summaries_a, clusters_a, summaries_b, clusters_b


def test_lmm_corrects_pseudoreplication_false_positive():
    # Seed chosen (from a small search) to land in the regime where naive
    # pooling of repeated per-cluster measurements manufactures a tiny
    # p-value out of what is, at the level of actual independent replicates
    # (clusters), a non-significant difference — the textbook
    # pseudoreplication failure mode. The cluster-aware LMM should not be
    # fooled by it.
    summaries_a, clusters_a, summaries_b, clusters_b = _clustered_dataset(seed=2)

    result = compare_groups(summaries_a, summaries_b, clusters_a=clusters_a, clusters_b=clusters_b)
    row = next(m for m in result["metrics"] if m["metric"] == "metric_x")

    assert row["p_value"] < 0.01  # naive Mann-Whitney U: fooled by pseudoreplication
    assert row["lmm_p_value"] > 0.05  # LMM: correctly not significant
    assert row["lmm_converged"] is True
    assert row["lmm_n_clusters"] == 8


def test_lmm_detects_genuine_group_effect_with_clusters():
    rng = np.random.default_rng(7)
    summaries_a, clusters_a, summaries_b, clusters_b = [], [], [], []
    for c in range(4):
        cluster_id = f"A{c}"
        for _ in range(10):
            summaries_a.append({"metric_x": 50 + rng.normal(0, 1.5)})
            clusters_a.append(cluster_id)
    for c in range(4):
        cluster_id = f"B{c}"
        for _ in range(10):
            summaries_b.append({"metric_x": 65 + rng.normal(0, 1.5)})
            clusters_b.append(cluster_id)

    result = compare_groups(summaries_a, summaries_b, clusters_a=clusters_a, clusters_b=clusters_b)
    row = next(m for m in result["metrics"] if m["metric"] == "metric_x")
    assert row["lmm_p_value"] < 0.01
    assert row["lmm_coefficient"] > 0  # group B is higher, matches the generating model
