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
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean frame-to-frame intensity change")
    ax.set_title(f"Beating signal — {result.summary.get('n_beats', 0)} beats detected")
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

    ax.set_title(f"{result.mode.upper()} segmentation — {result.n_objects} objects")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
