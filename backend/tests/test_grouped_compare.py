import numpy as np

from app.analysis import plotting
from app.analysis.group_stats import GroupInput, compare_grouped


def _summaries(rng, n, bpm, amp):
    return [
        {"filename": f"v{i}.mp4", "mean_bpm": float(rng.normal(bpm, 2)), "mean_amplitude": float(rng.normal(amp, 0.5))}
        for i in range(n)
    ]


def _design(rng):
    groups, cats = [], []
    for line, base in [("DC1", 40), ("DC2", 40), ("UCM1", 45)]:
        groups.append(GroupInput("Vehicle", _summaries(rng, 4, base, 10)))
        cats.append(line)
        groups.append(GroupInput("UT H", _summaries(rng, 4, base - 12, 6)))
        cats.append(line)
    return groups, cats


def test_compare_grouped_compares_conditions_within_each_category():
    rng = np.random.default_rng(3)
    groups, cats = _design(rng)
    comp = compare_grouped(groups, cats)
    assert comp["layout"] == "clustered"
    assert comp["categories"] == ["DC1", "DC2", "UCM1"]
    assert comp["conditions"] == ["Vehicle", "UT H"]
    assert len(comp["per_category"]) == 3
    for pc in comp["per_category"]:
        c = pc["comparison"]
        assert c["labels"] == ["Vehicle", "UT H"]
        bpm = next(m for m in c["metrics"] if m["metric"] == "mean_bpm")
        assert bpm["test"] == "mann_whitney_u"
        assert bpm["p_value"] is not None and bpm["p_value"] < 0.05
    # Flat metrics view carries every category·condition bar.
    flat = next(m for m in comp["metrics"] if m["metric"] == "mean_bpm")
    assert [g["label"] for g in flat["groups"]] == [
        "DC1 · Vehicle", "DC1 · UT H", "DC2 · Vehicle", "DC2 · UT H", "UCM1 · Vehicle", "UCM1 · UT H"
    ]
    assert comp["metric_keys"][0] in ("mean_bpm", "mean_amplitude")


def test_compare_grouped_single_condition_category_has_no_test():
    rng = np.random.default_rng(4)
    groups = [GroupInput("Vehicle", _summaries(rng, 3, 40, 10)), GroupInput("UT H", _summaries(rng, 3, 30, 6)),
              GroupInput("Vehicle", _summaries(rng, 3, 42, 9))]
    comp = compare_grouped(groups, ["DC1", "DC1", "DC2"])
    dc2 = comp["per_category"][1]["comparison"]
    assert dc2["labels"] == ["Vehicle"]
    assert all(m["test"] is None and m["p_value"] is None for m in dc2["metrics"])


def test_plot_clustered_comparison_writes_png_and_svg(tmp_path):
    rng = np.random.default_rng(5)
    groups, cats = _design(rng)
    comp = compare_grouped(groups, cats)
    comp["error_bar"] = "sd"
    comp["p_style"] = "nejm"
    comp["bracket_style"] = "line"
    comp["condition_colors"] = {"Vehicle": "#1f3fff", "UT H": "#ff3b30"}
    out = tmp_path / "plot.png"
    plotting.plot_group_comparison(comp, str(out))
    assert out.exists() and (tmp_path / "plot.svg").exists()
    # Flat layout still works with the bracket style switch.
    from app.analysis.group_stats import compare_groups
    flat = compare_groups(groups[:2], test_family="parametric")
    flat["bracket_style"] = "bracket"
    plotting.plot_group_comparison(flat, str(tmp_path / "flat.png"))
    assert (tmp_path / "flat.png").exists()


def test_compare_grouped_reports_two_way_anova():
    rng = np.random.default_rng(6)
    groups, cats = _design(rng)  # 3 lines x (Vehicle, UT H), 4 videos each
    comp = compare_grouped(groups, cats)
    tw = comp["two_way_anova"]
    assert "mean_bpm" in tw
    r = tw["mean_bpm"]
    assert r["sum_sq_type"] == "II"
    assert r["n"] == 24 and r["df_residual"] == 24 - 6
    # Strong treatment effect, no interaction built into the synthetic design.
    assert r["p_condition"] is not None and r["p_condition"] < 0.001
    assert r["p_interaction"] is not None and r["p_interaction"] > 0.05
    assert 0 <= r["p_category"] <= 1


def test_two_way_anova_skipped_when_a_cell_is_missing():
    rng = np.random.default_rng(8)
    groups = [GroupInput("Vehicle", _summaries(rng, 3, 40, 10)), GroupInput("UT H", _summaries(rng, 3, 30, 6)),
              GroupInput("Vehicle", _summaries(rng, 3, 42, 9))]
    comp = compare_grouped(groups, ["DC1", "DC1", "DC2"])
    assert comp["two_way_anova"] == {}


def test_individual_panels_are_written_for_both_layouts(tmp_path):
    rng = np.random.default_rng(9)
    groups, cats = _design(rng)
    comp = compare_grouped(groups, cats)
    comp["error_bar"] = "sem"
    panels = plotting.plot_group_comparison(comp, str(tmp_path / "plot.png"), panels_dir=str(tmp_path))
    assert len(panels) == len(comp["metric_keys"])
    for p in panels:
        assert (tmp_path / p["png"]).exists() and (tmp_path / p["svg"]).exists()
        assert p["png"].startswith("panel_")
    from app.analysis.group_stats import compare_groups
    flat = compare_groups(groups[:2])
    sub = tmp_path / "sub"; sub.mkdir()
    flat_panels = plotting.plot_group_comparison(flat, str(tmp_path / "flat.png"), panels_dir=str(sub))
    assert len(flat_panels) == len(flat["metrics"])
    assert all((sub / p["png"]).exists() for p in flat_panels)
    # Without panels_dir nothing extra is written and the return is empty.
    assert plotting.plot_group_comparison(flat, str(tmp_path / "flat2.png")) == []


def test_metric_missing_in_one_group_is_kept_with_n_zero(tmp_path):
    from app.analysis.group_stats import compare_groups

    rng = np.random.default_rng(11)
    a = _summaries(rng, 4, 40, 10)
    b = _summaries(rng, 4, 30, 6)
    c = [{"filename": f"c{i}.mp4", "mean_bpm": 0.0, "mean_amplitude": None} for i in range(3)]  # near-arrest
    comp = compare_groups([GroupInput("5.1 mM", a), GroupInput("9.0 mM", b), GroupInput("11.0 mM", c)])
    amp = next(m for m in comp["metrics"] if m["metric"] == "mean_amplitude")
    assert [g["n"] for g in amp["groups"]] == [4, 4, 0]
    assert amp["groups"][2]["mean"] is None
    assert amp["n_groups_with_data"] == 2
    assert amp["test"] == "mann_whitney_u" and amp["p_value"] is not None  # tested on the two groups with data
    bpm = next(m for m in comp["metrics"] if m["metric"] == "mean_bpm")
    assert bpm["groups"][2]["mean"] == 0.0  # 0 BPM is a real value and stays in the comparison
    # Plots must cope with the empty group.
    comp["error_bar"] = "sem"
    plotting.plot_group_comparison(comp, str(tmp_path / "plot.png"), panels_dir=str(tmp_path))
    assert (tmp_path / "plot.png").exists()
    grouped = compare_grouped(
        [GroupInput("Vehicle", a), GroupInput("UT H", c), GroupInput("Vehicle", b), GroupInput("UT H", b)],
        ["C6", "C6", "AR05", "AR05"],
    )
    grouped["error_bar"] = "sem"
    plotting.plot_group_comparison(grouped, str(tmp_path / "grouped.png"), panels_dir=str(tmp_path))
    assert (tmp_path / "grouped.png").exists()
