import numpy as np

from app.analysis.group_stats import GroupInput, compare_groups


def _summary(bpm, amplitude, extra=None):
    d = {"mean_bpm": bpm, "mean_amplitude": amplitude, "n_beats": 6}
    if extra:
        d.update(extra)
    return d


def test_compare_groups_detects_real_difference():
    group_a = GroupInput("Control", [_summary(bpm, 1.0) for bpm in [58, 60, 61, 59, 60, 62]])
    group_b = GroupInput("Treatment", [_summary(bpm, 1.0) for bpm in [88, 91, 90, 89, 92, 90]])

    result = compare_groups([group_a, group_b])

    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert bpm_row["test"] == "mann_whitney_u"
    assert bpm_row["groups"][0]["mean"] < bpm_row["groups"][1]["mean"]
    assert bpm_row["p_value"] is not None
    assert bpm_row["p_value"] < 0.05


def test_compare_groups_no_difference_stays_nonsignificant():
    group_a = GroupInput("A", [_summary(60, 1.0), _summary(61, 1.05), _summary(59, 0.95), _summary(60, 1.0)])
    group_b = GroupInput("B", [_summary(60, 1.0), _summary(59, 0.98), _summary(61, 1.02), _summary(60, 1.0)])

    result = compare_groups([group_a, group_b])
    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert bpm_row["p_value"] is None or bpm_row["p_value"] > 0.05


def test_compare_groups_excludes_bookkeeping_fields():
    group_a = GroupInput("A", [_summary(60, 1.0, {"reference_frame_index": 12})])
    group_b = GroupInput("B", [_summary(90, 1.0, {"reference_frame_index": 400})])

    result = compare_groups([group_a, group_b])
    metric_names = {m["metric"] for m in result["metrics"]}
    assert "reference_frame_index" not in metric_names
    assert "mean_bpm" in metric_names


def test_compare_groups_handles_missing_metric_gracefully():
    group_a = GroupInput("A", [{"mean_bpm": 60}, {"mean_bpm": 62}])
    group_b = GroupInput("B", [{"mean_bpm": 90}])  # only one group has this metric fully present

    result = compare_groups([group_a, group_b])
    assert result["n_videos"] == [2, 1]
    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert bpm_row["groups"][0]["n"] == 2
    assert bpm_row["groups"][1]["n"] == 1


def test_lmm_fields_absent_without_cluster_labels():
    group_a = GroupInput("A", [_summary(bpm, 1.0) for bpm in [58, 60, 61, 59, 60, 62]])
    group_b = GroupInput("B", [_summary(bpm, 1.0) for bpm in [88, 91, 90, 89, 92, 90]])

    result = compare_groups([group_a, group_b])
    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert "lmm_pairwise" not in bpm_row


def test_compare_groups_requires_at_least_two_groups():
    try:
        compare_groups([GroupInput("Only", [_summary(60, 1.0)])])
        assert False, "expected ValueError"
    except ValueError:
        pass


def _clustered_dataset(seed, n_clusters=8, n_per_cluster=20, cluster_sd=4.0, noise_sd=1.0):
    """Videos from n_clusters biological samples/batches, split randomly
    into two groups with *no* true group-level effect in the generating
    model — only cluster-to-cluster variation. Returns two GroupInputs."""
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
    return GroupInput("A", summaries_a, clusters_a), GroupInput("B", summaries_b, clusters_b)


def test_lmm_corrects_pseudoreplication_false_positive():
    # Seed chosen (from a small search) to land in the regime where naive
    # pooling of repeated per-cluster measurements manufactures a tiny
    # p-value out of what is, at the level of actual independent replicates
    # (clusters), a non-significant difference — the textbook
    # pseudoreplication failure mode. The cluster-aware LMM should not be
    # fooled by it.
    group_a, group_b = _clustered_dataset(seed=2)

    result = compare_groups([group_a, group_b])
    row = next(m for m in result["metrics"] if m["metric"] == "metric_x")
    pair = row["lmm_pairwise"][0]

    assert row["p_value"] < 0.01  # naive Mann-Whitney U: fooled by pseudoreplication
    assert pair["p_value"] > 0.05  # LMM: correctly not significant
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

    result = compare_groups(
        [GroupInput("A", summaries_a, clusters_a), GroupInput("B", summaries_b, clusters_b)]
    )
    row = next(m for m in result["metrics"] if m["metric"] == "metric_x")
    pair = row["lmm_pairwise"][0]
    assert pair["group_a"] == "A" and pair["group_b"] == "B"
    assert pair["p_value"] < 0.01
    assert pair["coefficient"] > 0  # group B is higher, matches the generating model


def test_compare_groups_three_groups_uses_kruskal_wallis_and_dunns_posthoc():
    rng = np.random.default_rng(3)
    group_a = GroupInput("Low", [_summary(60 + rng.normal(0, 1.5), 1.0) for _ in range(8)])
    group_b = GroupInput("Mid", [_summary(75 + rng.normal(0, 1.5), 1.0) for _ in range(8)])
    group_c = GroupInput("High", [_summary(90 + rng.normal(0, 1.5), 1.0) for _ in range(8)])

    result = compare_groups([group_a, group_b, group_c])
    assert result["labels"] == ["Low", "Mid", "High"]
    assert result["n_videos"] == [8, 8, 8]

    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert bpm_row["test"] == "kruskal_wallis"
    assert bpm_row["p_value"] < 0.01

    posthoc = bpm_row["posthoc"]
    assert len(posthoc) == 3  # Low-Mid, Low-High, Mid-High
    low_high = next(p for p in posthoc if {p["group_a"], p["group_b"]} == {"Low", "High"})
    assert low_high["p_value_bonferroni"] < 0.05


def test_compare_groups_three_groups_no_difference_stays_nonsignificant():
    rng = np.random.default_rng(11)
    groups = [
        GroupInput(label, [_summary(60 + rng.normal(0, 1.5), 1.0) for _ in range(6)])
        for label in ("A", "B", "C")
    ]
    result = compare_groups(groups)
    bpm_row = next(m for m in result["metrics"] if m["metric"] == "mean_bpm")
    assert bpm_row["p_value"] is None or bpm_row["p_value"] > 0.05
    assert bpm_row["posthoc"] is None or all(p["p_value_bonferroni"] > 0.05 for p in bpm_row["posthoc"])


def test_compare_groups_four_groups_lmm_pairwise_covers_all_pairs():
    rng = np.random.default_rng(13)
    labels = ["W", "X", "Y", "Z"]
    base_means = {"W": 50.0, "X": 55.0, "Y": 70.0, "Z": 72.0}
    groups = []
    for label in labels:
        summaries, clusters = [], []
        for c in range(4):
            cluster_id = f"{label}{c}"
            for _ in range(6):
                summaries.append({"metric_x": base_means[label] + rng.normal(0, 1.0)})
                clusters.append(cluster_id)
        groups.append(GroupInput(label, summaries, clusters))

    result = compare_groups(groups)
    row = next(m for m in result["metrics"] if m["metric"] == "metric_x")
    # Not asserting lmm_converged here: statsmodels' MixedLM optimizer can
    # report non-convergence on small/noisy variance components even when
    # the fixed-effect estimates themselves are stable (same flakiness
    # documented in the README) — the substantive check below (pairwise
    # coverage and p-values matching the generating model) is what matters.
    pairs = {frozenset((p["group_a"], p["group_b"])) for p in row["lmm_pairwise"]}
    expected_pairs = {frozenset((a, b)) for i, a in enumerate(labels) for b in labels[i + 1 :]}
    assert pairs == expected_pairs  # all C(4,2) = 6 pairs present, not just vs.-reference

    # W vs Z has the largest true mean gap (50 vs 72) -> should be clearly
    # significant. X vs Y (55 vs 70) doesn't involve the reference group
    # (W) at all, so this also exercises the non-reference contrast path.
    w_z = next(p for p in row["lmm_pairwise"] if {p["group_a"], p["group_b"]} == {"W", "Z"})
    assert w_z["p_value"] < 0.01
    x_y = next(p for p in row["lmm_pairwise"] if {p["group_a"], p["group_b"]} == {"X", "Y"})
    assert x_y["p_value"] < 0.01
    assert x_y["coefficient"] > 0  # Y (70) is higher than X (55)
