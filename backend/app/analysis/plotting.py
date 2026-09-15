"""Matplotlib figure generation for each analysis type.

Kept separate from the numeric analysis code so the analysis functions stay
pure/testable and plotting (which needs a display-less backend) is isolated.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from app.analysis.beating import BeatingResult
from app.analysis.calcium import CalciumResult
from app.analysis.morphology import MorphologyResult


def plot_beating(result: BeatingResult, out_path: str) -> None:
    has_piv_field = result.piv_field is not None
    if has_piv_field:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    else:
        fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(result.time_s, result.smoothed_signal, color="#c0392b", linewidth=1.3, label="motion signal")
    if len(result.peak_indices):
        ax.plot(
            result.time_s[result.peak_indices],
            result.smoothed_signal[result.peak_indices],
            "o",
            color="#2c3e50",
            markersize=5,
            label="beat peak",
        )
    if len(result.trough_indices):
        ax.plot(
            result.time_s[result.trough_indices],
            result.smoothed_signal[result.trough_indices],
            "v",
            color="#2980b9",
            markersize=4,
            label="baseline",
        )
    ylabels = {
        "reference": "Mean |frame − reference frame| intensity",
        "consecutive": "Mean frame-to-frame intensity change",
        "piv": "Mean PIV displacement magnitude (px)",
    }
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabels.get(result.signal_mode, "Motion signal"))
    ax.set_title(f"Beating signal ({result.signal_mode}) — {result.summary.get('n_beats', 0)} beats detected")
    ax.legend(loc="upper right", fontsize=8)

    if has_piv_field:
        _draw_piv_field(ax2, result.piv_field)
        ax2.set_title(f"PIV vector field @ frame {result.piv_field['frame_index']} (strongest beat)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_piv_field(ax, field: dict) -> None:
    """Vector arrows over a magnitude heatmap — the standard PIV output
    visualization (matches PIVlab/PIV-MyoMonitor's vector-arrow + heatmap
    figures)."""
    x, y, u, v = field["x"], field["y"], field["u"], field["v"]
    magnitude = np.sqrt(u**2 + v**2)
    im = ax.imshow(
        magnitude,
        extent=(x.min(), x.max(), y.max(), y.min()),
        cmap="viridis",
        aspect="auto",
        alpha=0.85,
    )
    ax.quiver(x, y, u, v, color="white", scale_units="xy", angles="xy", width=0.004)
    ax.figure.colorbar(im, ax=ax, label="displacement magnitude (px)", fraction=0.046, pad=0.04)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()


def plot_calcium(result: CalciumResult, out_path: str) -> None:
    """dF/F0 trace with detected transients, plus (when available) the
    pixel-wise peak dF/F0 map showing where in the field the signal is."""
    has_map = getattr(result, "df_f0_map", None) is not None
    if has_map:
        fig, (ax, ax_map) = plt.subplots(
            1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2.2, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax_map = None

    ax.plot(result.time_s, result.df_f0, color="#16a085", linewidth=1.3, label="dF/F0")
    if len(result.peak_indices):
        ax.plot(
            result.time_s[result.peak_indices],
            result.df_f0[result.peak_indices],
            "o",
            color="#2c3e50",
            markersize=5,
            label="transient peak",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF / F0")
    bg = result.summary.get("background_method", "none")
    bg_note = "" if bg == "none" else f" (background subtracted: {bg})"
    ax.set_title(f"Calcium transients — {result.summary.get('n_transients', 0)} detected{bg_note}")
    ax.legend(loc="upper right", fontsize=8)

    if ax_map is not None:
        im = ax_map.imshow(result.df_f0_map, cmap="inferno", vmin=0)
        ax_map.set_title("Pixel-wise peak ΔF/F0", fontsize=9)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label="peak ΔF/F0")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_morphology(result: MorphologyResult, out_path: str) -> None:
    has_texture_map = result.orientation_map is not None
    if has_texture_map:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, ax = plt.subplots(figsize=(6, 6))

    proj = result.projection.astype(float)
    proj_norm = (proj - proj.min()) / (np.ptp(proj) + 1e-9)
    ax.imshow(proj_norm, cmap="gray")

    boundaries = find_boundaries(result.label_image, mode="outer")
    overlay = np.zeros((*boundaries.shape, 4))
    overlay[boundaries] = [1.0, 0.85, 0.0, 1.0]
    ax.imshow(overlay)

    df = result.objects_df
    if result.mode == "2d" and {"centroid-0", "centroid-1", "orientation", "major_axis_length"} <= set(df.columns):
        for _, row in df.iterrows():
            y0, x0 = row["centroid-0"], row["centroid-1"]
            half_len = row["major_axis_length"] / 2.0
            angle = row["orientation"]
            dx, dy = np.cos(angle) * half_len, -np.sin(angle) * half_len
            ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], "-", color="#00e5ff", linewidth=1.5)
    elif result.mode == "3d" and {"centroid-1", "centroid-2", "axis_y", "axis_x", "equivalent_diameter_area"} <= set(
        df.columns
    ):
        for _, row in df.iterrows():
            y0, x0 = row["centroid-1"], row["centroid-2"]
            half_len = row["equivalent_diameter_area"] / 2.0
            dx, dy = row["axis_x"] * half_len, row["axis_y"] * half_len
            ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], "-", color="#00e5ff", linewidth=1.5)

    alignment = result.summary.get("alignment_score", result.summary.get("alignment_score_3d"))
    title = f"{result.mode.upper()} segmentation — {result.n_objects} objects"
    if alignment is not None:
        title += f", alignment {alignment:.2f}"
    ax.set_title(title)
    ax.axis("off")

    if has_texture_map:
        _draw_orientation_map(ax2, result.orientation_map, result.coherence_map)
        st_score = result.summary.get("texture_alignment_score")
        ax2.set_title(f"Structure-tensor orientation, alignment {st_score:.2f}" if st_score is not None else "Structure-tensor orientation")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_orientation_map(ax, orientation_map: np.ndarray, coherence_map: np.ndarray) -> None:
    """HSV-encoded local-orientation map: hue = angle, value = coherence.

    Standard visualization for structure-tensor / OrientationJ-style fiber
    orientation maps — hue cycles once over the axial [-pi/2, pi/2] range
    (so opposite-hue colors mean perpendicular, not just "different"), and
    low-coherence (unreliable / isotropic) regions fade to black instead of
    showing an arbitrary color.
    """
    from matplotlib.colors import hsv_to_rgb

    hue = (orientation_map + np.pi / 2) / np.pi
    coherence_norm = coherence_map / (coherence_map.max() + 1e-9)
    hsv = np.stack([hue, np.ones_like(hue), coherence_norm], axis=-1)
    ax.imshow(hsv_to_rgb(hsv))
    ax.axis("off")


def _format_p(p: float | None) -> str:
    """GraphPad Prism-style p-value text: exact to 4 decimals, '< 0.0001' below that."""
    if p is None:
        return "p = n/a"
    if p < 0.0001:
        return "p < 0.0001"
    return f"p = {p:.4f}"


def _bracket_pairs(metric: dict, labels: list[str]) -> list[tuple[int, int, float]]:
    """(i, j, p) comparisons to draw as significance brackets for one metric.

    Mirrors the usual dose-response / treatment-series figure layout: every
    group is compared against the first group (the control/reference — the
    first group the user adds), not all-vs-all, which for 5 groups would be
    10 brackets and unreadable. All-pairs p-values are still in the results
    table and summary.json. p-values shown are each post-hoc entry's
    `p_adjusted` (Dunn's/Bonferroni or Tukey HSD, per test family) for 3+
    groups, or the two-group test's p (Mann-Whitney U or Welch's t) for 2.
    Sorted so the shortest bracket sits lowest and the longest on top.
    """
    if metric["test"] in ("mann_whitney_u", "welch_t"):
        return [(0, 1, metric["p_value"])] if metric["p_value"] is not None else []

    posthoc = metric.get("posthoc") or []
    reference = labels[0]
    pairs = []
    for entry in posthoc:
        if entry["group_a"] != reference or entry["group_b"] not in labels:
            continue
        p_adj = entry.get("p_adjusted")
        if p_adj is None:
            continue
        pairs.append((0, labels.index(entry["group_b"]), p_adj))
    pairs.sort(key=lambda t: t[1] - t[0])
    return pairs


_TEST_LABELS = {
    "mann_whitney_u": "Mann-Whitney U",
    "kruskal_wallis": "Kruskal-Wallis",
    "welch_t": "Welch's t-test",
    "anova": "One-way ANOVA",
}


# Control in dark gray, then colors close to the Prism defaults the user's
# lab figures use (pink / teal / purple / lavender), then fallbacks.
_GROUP_COLORS = [
    "#595959", "#ec4b81", "#2a9d8f", "#5b3a9b", "#b39ddb",
    "#e67e22", "#3498db", "#27ae60", "#c0392b", "#7f8c8d",
]


def plot_group_comparison(comparison: dict, out_path: str, max_metrics: int = 12) -> None:
    """One GraphPad-Prism-style panel per metric, across all groups.

    Each panel: bar = group mean, error bar = SEM, overlaid dots = each
    video's value, and stacked significance brackets with p-values for
    each group vs. the first (control/reference) group — the layout of a
    typical dose-response figure. Panels are ordered most-significant-first
    (comparison["metrics"] is already sorted that way) and capped at
    max_metrics so a summary with many fields doesn't produce an
    unreadably large grid. Works for any number of groups (2+) — the
    omnibus test in each title is Mann-Whitney U for exactly 2 groups or
    Kruskal-Wallis for 3+, per compare_groups().
    """
    metrics = comparison["metrics"][:max_metrics]
    error_bar = comparison.get("error_bar", "sem")
    if error_bar not in ("sem", "sd"):
        error_bar = "sem"

    if not metrics:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No comparable numeric metrics", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.9 * nrows), squeeze=False)
    rng = np.random.default_rng(0)

    for idx, m in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        groups = m["groups"]
        labels = [g["label"] for g in groups]
        n_groups = len(groups)
        xs = np.arange(n_groups)
        means = [g["mean"] for g in groups]
        # Error bar = SEM or SD per comparison["error_bar"]: SEM is the usual
        # choice in cardiomyocyte/organoid figures, SD what the lab's own
        # manuscripts report ("mean ± SD") — both are valid, they must just
        # be labeled, so the y-axis says which one this is.
        if error_bar == "sd":
            errs = [g["std"] for g in groups]
        else:
            errs = [g.get("sem", g["std"] / np.sqrt(g["n"]) if g["n"] > 1 else 0.0) for g in groups]
        sems = errs
        colors = [_GROUP_COLORS[i % len(_GROUP_COLORS)] for i in range(n_groups)]

        ax.bar(xs, means, width=0.62, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8, zorder=2)
        ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=4)

        all_values: list[float] = []
        for gi, g in enumerate(groups):
            jitter = rng.normal(0, 0.06, len(g["values"]))
            ax.scatter(
                xs[gi] + jitter, g["values"], color=colors[gi], edgecolor="black", linewidth=0.6, s=30, zorder=5
            )
            all_values.extend(g["values"])

        data_max = max(max(all_values, default=0.0), max(mu + s for mu, s in zip(means, sems)))
        data_min = min(min(all_values, default=0.0), 0.0)
        span = (data_max - data_min) or 1.0

        # Stacked brackets above the data, shortest lowest — same look as
        # Prism's "compare to control" annotations.
        height = data_max + 0.07 * span
        step = 0.12 * span
        tick = 0.025 * span
        pairs = _bracket_pairs(m, labels)
        for i, j, p in pairs:
            ax.plot(
                [xs[i], xs[i], xs[j], xs[j]],
                [height - tick, height, height, height - tick],
                color="black",
                linewidth=1.0,
                zorder=6,
            )
            ax.text((xs[i] + xs[j]) / 2, height + 0.012 * span, _format_p(p), ha="center", va="bottom", fontsize=7.5)
            height += step
        top = height + 0.03 * span if pairs else data_max + 0.12 * span
        ax.set_ylim(data_min - 0.04 * span if data_min < 0 else 0.0, top)

        ax.set_xticks(xs)
        ax.set_xticklabels(
            labels,
            fontsize=8,
            rotation=30 if n_groups > 3 else 0,
            ha="right" if n_groups > 3 else "center",
        )
        ax.set_xlim(-0.6, n_groups - 0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(f"mean ± {error_bar.upper()}", fontsize=8)

        test_label = _TEST_LABELS.get(m["test"], m["test"])
        title_extra = ""
        lmm_pairwise = m.get("lmm_pairwise")
        if lmm_pairwise:
            n_sig_lmm = sum(1 for pw in lmm_pairwise if pw["p_value"] < 0.05)
            title_extra = f"\nLMM: {n_sig_lmm}/{len(lmm_pairwise)} pairs sig ({m.get('lmm_n_clusters')} clusters)"
        ax.set_title(f"{m['metric']}\n{test_label} {_format_p(m['p_value'])}{title_extra}", fontsize=9)

    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    n_groups_total = len(comparison["labels"])
    posthoc_label = comparison.get("posthoc_label", "post-hoc")
    bracket_note = (
        f"Brackets: {posthoc_label} p-value"
        if n_groups_total == 2
        else f"Brackets: each group vs. '{comparison['labels'][0]}', {posthoc_label} p"
    )
    fig.text(0.5, 0.005, bracket_note, ha="center", va="bottom", fontsize=8, color="#444444")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_agreement(agreement: dict, label_a: str, label_b: str, out_path: str) -> None:
    """Two-panel method-agreement figure: scatter (with identity + regression
    lines) and Bland-Altman, the standard pairing for a validation-study
    figure in the biomedical literature."""
    a = np.array(agreement["values_a"])
    b = np.array(agreement["values_b"])
    diffs = np.array(agreement["diffs"])
    means = np.array(agreement["means"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    pad = (hi - lo) * 0.08 if hi > lo else 1.0
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#999999", linewidth=1, label="y = x")
    xs = np.linspace(lo - pad, hi + pad, 50)
    ys = agreement["regression_slope"] * xs + agreement["regression_intercept"]
    ax1.plot(xs, ys, "-", color="#e67e22", linewidth=1.5, label="regression")
    ax1.scatter(a, b, color="#3498db", alpha=0.75, s=25, zorder=3)
    ax1.set_xlabel(label_a)
    ax1.set_ylabel(label_b)
    ax1.set_title(f"r={agreement['pearson_r']:.3f}, ICC={agreement['icc_2_1']:.3f}, n={agreement['n']}", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)

    bias = agreement["bland_altman_bias"]
    loa_lower = agreement["bland_altman_loa_lower"]
    loa_upper = agreement["bland_altman_loa_upper"]
    ax2.scatter(means, diffs, color="#3498db", alpha=0.75, s=25, zorder=3)
    ax2.axhline(bias, color="#2c3e50", linewidth=1.5, label=f"bias = {bias:.3g}")
    ax2.axhline(loa_upper, color="#c0392b", linestyle="--", linewidth=1.2, label="95% LoA")
    ax2.axhline(loa_lower, color="#c0392b", linestyle="--", linewidth=1.2)
    ax2.set_xlabel(f"Mean of {label_a} & {label_b}")
    ax2.set_ylabel(f"{label_a} − {label_b}")
    ax2.set_title("Bland-Altman", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_colocalization(
    channel_a: np.ndarray,
    channel_b: np.ndarray,
    stats: dict,
    label_a: str,
    label_b: str,
    out_path: str,
    max_scatter_points: int = 20000,
) -> None:
    """Standard colocalization figure: RGB merge (A=red, B=green, overlap=
    yellow) and the pixel-intensity scatter plot, side by side."""

    def norm(x: np.ndarray) -> np.ndarray:
        x = x.astype(float)
        rng = np.ptp(x)
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    a_norm, b_norm = norm(channel_a), norm(channel_b)
    merge = np.zeros((*a_norm.shape, 3))
    merge[..., 0] = a_norm
    merge[..., 1] = b_norm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax1.imshow(merge)
    ax1.set_title(f"{label_a} (red) / {label_b} (green) merge", fontsize=10)
    ax1.axis("off")

    a_flat, b_flat = channel_a.flatten().astype(float), channel_b.flatten().astype(float)
    if len(a_flat) > max_scatter_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(a_flat), max_scatter_points, replace=False)
        a_flat, b_flat = a_flat[idx], b_flat[idx]
    ax2.scatter(a_flat, b_flat, s=2, alpha=0.25, color="#3498db")
    ax2.set_xlabel(f"{label_a} intensity")
    ax2.set_ylabel(f"{label_b} intensity")
    r = stats["pearson_r"]
    m1, m2 = stats["manders_m1"], stats["manders_m2"]
    ax2.set_title(f"r={r:.3f}, M1={m1:.3f}, M2={m2:.3f}", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
