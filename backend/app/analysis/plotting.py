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
    ylabel = (
        "Mean |frame − reference frame| intensity"
        if result.signal_mode == "reference"
        else "Mean frame-to-frame intensity change"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Beating signal ({result.signal_mode}) — {result.summary.get('n_beats', 0)} beats detected")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calcium(result: CalciumResult, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
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
    ax.set_title(f"Calcium transients — {result.summary.get('n_transients', 0)} detected")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_morphology(result: MorphologyResult, out_path: str) -> None:
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_group_comparison(comparison: dict, out_path: str, max_metrics: int = 12) -> None:
    """One dot-plot-with-mean-bar panel per metric, group A vs group B.

    Panels are ordered most-significant-first (comparison["metrics"] is
    already sorted that way) and capped at max_metrics so a summary with
    many fields doesn't produce an unreadably large grid.
    """
    metrics = comparison["metrics"][:max_metrics]
    label_a, label_b = comparison["label_a"], comparison["label_b"]

    if not metrics:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No comparable numeric metrics", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for idx, m in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        rng = np.random.default_rng(0)
        xa = rng.normal(0, 0.05, len(m["values_a"]))
        xb = rng.normal(1, 0.05, len(m["values_b"]))
        ax.scatter(xa, m["values_a"], color="#3498db", alpha=0.8, s=25, zorder=3)
        ax.scatter(xb, m["values_b"], color="#e67e22", alpha=0.8, s=25, zorder=3)
        ax.errorbar(
            [0, 1],
            [m["mean_a"], m["mean_b"]],
            yerr=[m["std_a"], m["std_b"]],
            fmt="_",
            color="black",
            markersize=20,
            markeredgewidth=2,
            capsize=4,
            zorder=4,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_a, label_b], fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        p = m["p_value"]
        p_text = f"p={p:.3g}" if p is not None else "p=n/a"
        sig = " *" if (p is not None and p < 0.05) else ""
        ax.set_title(f"{m['metric']}\n{p_text}{sig}", fontsize=9)

    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
