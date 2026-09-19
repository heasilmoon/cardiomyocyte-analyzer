"""Synthetic accuracy benchmark for the beating (contraction) analysis.

Generates videos of a textured, radially contracting "tissue" with a
KNOWN beat schedule, runs analyze_beating() in each signal mode with the
app's default parameters, and scores how often the detected beat count and
BPM match the ground truth — across true BPM, frame rate, sensor noise and
texture contrast (the last mainly matters for PIV mode).

This is NOT a substitute for validation against a reference tool on real
recordings (see the README's validation workflow / the app's "검증" tab for
that). It answers a narrower question: under which recording conditions
does the detector itself start missing or over-counting beats, given
perfectly known truth? Real data adds things this can't model (focus
drift, debris, cells outside the field moving, uneven illumination).

Run from backend/:
    PYTHONPATH=. python benchmarks/beating_accuracy.py            # full grid
    PYTHONPATH=. python benchmarks/beating_accuracy.py --quick    # smoke test
Outputs go to benchmarks/results/ (CSV of every run, summary tables, heatmap).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from app.analysis.beating import analyze_beating

SIZE = 160
DURATION_S = 10.0
TISSUE_RADIUS = 60
CONTRACTION_FRACTION = 0.04  # 4 % radial shrink at peak contraction


def beat_schedule(bpm: float, duration_s: float, rng: np.random.Generator, jitter: float = 0.08) -> np.ndarray:
    """Beat peak times with ±jitter (fractional) inter-beat variability."""
    period = 60.0 / bpm
    times = []
    t = period * rng.uniform(0.6, 1.0)  # first beat not at t=0
    while t < duration_s - 0.15:
        times.append(t)
        t += period * (1.0 + rng.uniform(-jitter, jitter))
    return np.asarray(times)


def contraction_pulse(t: np.ndarray, beat_times: np.ndarray, bpm: float) -> np.ndarray:
    """0..1 contraction profile: fast rise, slower exponential relaxation."""
    period = 60.0 / bpm
    rise_sigma = min(0.06, 0.15 * period)
    relax_tau = min(0.15, 0.3 * period)
    p = np.zeros_like(t)
    for tb in beat_times:
        before = t < tb
        p += np.where(before, np.exp(-((t - tb) / rise_sigma) ** 2), np.exp(-(t - tb) / relax_tau))
    return np.clip(p, 0.0, 1.0)


def base_tissue_image(rng: np.random.Generator, texture_contrast: float) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    c = SIZE / 2
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    tissue = r <= TISSUE_RADIUS
    speckle = ndi.gaussian_filter(rng.standard_normal((SIZE, SIZE)), sigma=1.5)
    speckle = speckle / (np.abs(speckle).max() + 1e-9)
    img = np.full((SIZE, SIZE), 40.0)
    img[tissue] = 110.0 + texture_contrast * 70.0 * speckle[tissue]
    return img


def render_video(
    bpm: float, fps: float, noise_sd: float, texture_contrast: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(frames uint8 (N,H,W), true beat times). Tissue contracts radially about the center."""
    rng = np.random.default_rng(seed)
    n_frames = int(round(DURATION_S * fps))
    t = np.arange(n_frames) / fps
    beats = beat_schedule(bpm, DURATION_S, rng)
    pulse = contraction_pulse(t, beats, bpm)
    base = base_tissue_image(rng, texture_contrast)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    c = SIZE / 2
    frames = np.empty((n_frames, SIZE, SIZE), dtype=np.uint8)
    for i in range(n_frames):
        s = 1.0 - CONTRACTION_FRACTION * pulse[i]  # shrink factor (1 = resting)
        src_y = c + (yy - c) / s
        src_x = c + (xx - c) / s
        frame = ndi.map_coordinates(base, [src_y, src_x], order=1, mode="nearest")
        if noise_sd > 0:
            frame = frame + rng.normal(0.0, noise_sd, frame.shape)
        frames[i] = np.clip(frame, 0, 255).astype(np.uint8)
    return frames, beats


def score(result_summary: dict, beats: np.ndarray) -> dict:
    true_n = int(len(beats))
    det_n = int(result_summary["n_beats"])
    true_bpm = 60.0 / np.mean(np.diff(beats)) if len(beats) > 1 else None
    det_bpm = result_summary.get("mean_bpm")
    bpm_err_pct = (
        abs(det_bpm - true_bpm) / true_bpm * 100.0 if (true_bpm and det_bpm is not None) else None
    )
    return {
        "true_n_beats": true_n,
        "detected_n_beats": det_n,
        "exact": det_n == true_n,
        "within_1": abs(det_n - true_n) <= 1,
        "true_bpm": true_bpm,
        "detected_bpm": det_bpm,
        "bpm_abs_error_pct": bpm_err_pct,
    }


def run_grid(quick: bool) -> pd.DataFrame:
    bpms = [30, 60, 90, 120, 180]
    fpss = [10, 15, 30]
    noises = [0, 5, 10, 20]
    textures = [1.0, 0.25]
    seeds = [0, 1, 2]
    modes = ["reference", "consecutive", "piv", "optical_flow"]
    if quick:
        bpms, fpss, noises, textures, seeds = [60, 120], [30], [0, 10], [1.0], [0]

    rows = []
    combos = list(itertools.product(bpms, fpss, noises, textures, seeds))
    t0 = time.time()
    for k, (bpm, fps, noise, tex, seed) in enumerate(combos, 1):
        frames, beats = render_video(bpm, fps, noise, tex, seed)
        for mode in modes:
            try:
                res = analyze_beating(frames, float(fps), signal_mode=mode)
                sc = score(res.summary, beats)
                sc["error"] = None
            except Exception as exc:  # keep going; record the failure
                sc = {
                    "true_n_beats": len(beats),
                    "detected_n_beats": None,
                    "exact": False,
                    "within_1": False,
                    "true_bpm": None,
                    "detected_bpm": None,
                    "bpm_abs_error_pct": None,
                    "error": repr(exc),
                }
            rows.append(
                {"mode": mode, "bpm": bpm, "fps": fps, "noise_sd": noise, "texture": tex, "seed": seed, **sc}
            )
        if k % 10 == 0 or k == len(combos):
            print(f"  {k}/{len(combos)} videos done ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, out_dir: Path) -> str:
    lines = []

    def table(group_cols: list[str], title: str) -> None:
        g = df.groupby(group_cols).agg(
            exact_rate=("exact", "mean"),
            within_1_rate=("within_1", "mean"),
            bpm_err_pct=("bpm_abs_error_pct", "median"),
            n=("exact", "size"),
        )
        g["exact_rate"] = (g["exact_rate"] * 100).round(0).astype(int)
        g["within_1_rate"] = (g["within_1_rate"] * 100).round(0).astype(int)
        g["bpm_err_pct"] = g["bpm_err_pct"].round(1)
        lines.append(f"\n### {title}\n")
        lines.append(g.reset_index().to_markdown(index=False))

    table(["mode"], "Overall by signal mode (% of runs)")
    table(["mode", "bpm"], "By true BPM")
    table(["mode", "fps"], "By frame rate")
    table(["mode", "noise_sd"], "By sensor noise (std, 8-bit units)")
    table(["mode", "texture"], "By texture contrast (1.0 = strong speckle, 0.25 = faint)")

    text = "\n".join(lines)
    (out_dir / "summary.md").write_text(text)
    return text


def heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    modes = ["reference", "consecutive", "piv", "optical_flow"]
    fig, axes = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 4.2), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        sub = df[(df["mode"] == mode) & (df["fps"] == df["fps"].max()) & (df["texture"] == df["texture"].max())]
        pivot = sub.pivot_table(index="noise_sd", columns="bpm", values="exact", aggfunc="mean") * 100
        im = ax.imshow(pivot.values, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("true BPM")
        ax.set_ylabel("noise sd")
        ax.set_title(f"{mode}: exact beat-count match (%)\n@ {int(df['fps'].max())} fps, strong texture", fontsize=9)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
    fig.savefig(out_dir / "exact_match_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny grid, for a smoke test")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_grid(args.quick)
    df.to_csv(out_dir / ("runs_quick.csv" if args.quick else "runs.csv"), index=False)
    if not args.quick:
        print(summarize(df, out_dir))
        heatmap(df, out_dir)
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "n_runs": int(len(df)),
                    "size_px": SIZE,
                    "duration_s": DURATION_S,
                    "contraction_fraction": CONTRACTION_FRACTION,
                    "default_params": "analyze_beating defaults (prominence_frac=0.15, auto min_bpm_gap)",
                },
                indent=2,
            )
        )
    else:
        print(df[["mode", "bpm", "noise_sd", "true_n_beats", "detected_n_beats", "bpm_abs_error_pct", "error"]])


if __name__ == "__main__":
    main()
