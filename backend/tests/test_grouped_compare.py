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
