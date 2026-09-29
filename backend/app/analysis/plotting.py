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

    is_flow = result.signal_mode == "optical_flow"
    ax.plot(
        result.time_s,
        result.smoothed_signal,
        color="#c0392b",
        linewidth=1.3,
        label="average speed" if is_flow else "motion signal",
    )
    if len(result.peak_indices):
        ax.plot(
            result.time_s[result.peak_indices],
            result.smoothed_signal[result.peak_indices],
            "o",
            color="#2c3e50",
            markersize=5,
            label="contraction peak (MCS)" if is_flow else "beat peak",
        )
    secondary = getattr(result, "secondary_peak_indices", None)
    if secondary is not None and len(secondary):
        ax.plot(
            result.time_s[secondary],
            result.smoothed_signal[secondary],
            "s",
            color="#8e44ad",
            markersize=5,
            label="relaxation peak (MRS)",
        )
    if len(result.trough_indices):
        ax.plot(
            result.time_s[result.trough_indices],
            result.smoothed_signal[result.trough_indices],
            "v",
            color="#2980b9",
            markersize=4,
            label="wave start" if is_flow else "baseline",
        )
    wave_ends = getattr(result, "wave_end_indices", None)
    if wave_ends is not None and len(wave_ends):
        ax.plot(
            result.time_s[wave_ends],
            result.smoothed_signal[wave_ends],
            "^",
            color="#16a085",
            markersize=4,
            label="wave end",
        )
    if is_flow and result.summary.get("wave_threshold_frac") is not None:
        base = result.summary.get("baseline_speed")
        if base is not None:
            ax.axhline(base, color="#7f8c8d", linewidth=0.8, linestyle=":", label="baseline")
    units = getattr(result, "signal_units", None) or "px/s"
    ylabels = {
        "reference": "Mean |frame − reference frame| intensity",
        "consecutive": "Mean frame-to-frame intensity change",
        "piv": "Mean PIV displacement magnitude (px)",
        "optical_flow": f"Average optical-flow speed ({units.replace('um', 'µm')})",
    }
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabels.get(result.signal_mode, "Motion signal"))
    ax.set_title(f"Beating signal ({result.signal_mode}) — {result.summary.get('n_beats', 0)} beats detected")
    ax.legend(loc="upper right", fontsize=8)

    if has_piv_field:
        field = result.piv_field
        if field.get("kind") == "optical_flow":
            _draw_piv_field(ax2, field, label=f"speed ({units.replace('um', 'µm')})")
            ax2.set_title(
                f"Farneback optical-flow field @ frame {field['frame_index']} (strongest contraction)", fontsize=10
            )
        else:
            _draw_piv_field(ax2, field)
            ax2.set_title(f"PIV vector field @ frame {field['frame_index']} (strongest beat)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_piv_field(ax, field: dict, label: str = "displacement magnitude (px)") -> None:
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
    ax.figure.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
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


def _nejm_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _short_p(p: float | None) -> str:
    """Compact p for crowded subtitles: 'p < 0.001' or 'p = 0.034'."""
    if p is None:
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def _format_p(p: float | None, style: str = "nejm") -> str:
    """p-value text for brackets/subtitles.

    style "nejm" (default, Prism's "NEJM" P value style): two significant
    digits with the significance level in parentheses — p = 0.12 (ns),
    p = 0.033 (*), p = 0.002 (**), p < 0.001 (***).
    style "value": Prism's exact-p style (4 decimals, '< 0.0001').
    style "stars": asterisks only (* < 0.05, ** < 0.01, *** < 0.001,
    **** < 0.0001, 'ns' otherwise).
    """
    if p is None:
        return "n/a" if style == "stars" else "p = n/a"
    if style == "nejm":
        if p < 0.001:
            return "p < 0.001 (***)"
        return f"p = {p:.2g} ({_nejm_stars(p)})"
    if style == "stars":
        if p < 0.0001:
            return "****"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"
    if p < 0.0001:
        return "p < 0.0001"
    return f"p = {p:.4f}"


# Publication display names: (title, y-axis label). Keys are the summary
# fields of the beating / calcium / morphology analyses. Anything not listed
# is prettified from its key by _metric_display().
_METRIC_DISPLAY: dict[str, tuple[str, str]] = {
    # Beating (reference / consecutive / piv)
    "n_beats": ("Number of beats", "count"),
    "mean_bpm": ("Beating rate", "BPM"),
    "mean_inter_beat_interval_s": ("Inter-beat interval", "s"),
    "ibi_std_s": ("Inter-beat interval SD", "s"),
    "ibi_cv_percent": ("Beating rate variability", "IBI CV (%)"),
    "mean_amplitude": ("Contraction amplitude", "a.u."),
    "amplitude_cv_percent": ("Amplitude variability", "CV (%)"),
    "mean_contraction_time_s": ("Contraction time", "s"),
    "mean_relaxation_time_s": ("Relaxation time", "s"),
    "mean_max_contraction_velocity": ("Max. contraction velocity", "a.u./s"),
    "mean_max_relaxation_velocity": ("Max. relaxation velocity", "a.u./s"),
    "mean_time_to_decay_10_s": ("Time to 10% relaxation", "s"),
    "mean_time_to_decay_50_s": ("Time to 50% relaxation", "s"),
    "mean_time_to_decay_90_s": ("Time to 90% relaxation", "s"),
    "estimated_period_s": ("Estimated beat period", "s"),
    "duration_s": ("Recording length", "s"),
    # Beating (optical_flow, ContractionWave parameters)
    "n_complete_waves": ("Complete contraction–relaxation waves", "count"),
    "baseline_speed": ("Baseline speed", "speed"),
    "mean_max_contraction_speed": ("Max. contraction speed (MCS)", "speed"),
    "mean_max_relaxation_speed": ("Max. relaxation speed (MRS)", "speed"),
    "mean_mcs_mrs_difference": ("MCS − MRS", "speed"),
    "max_max_contraction_speed": ("Peak MCS", "speed"),
    "mean_contraction_time_to_peak_s": ("Contraction time-to-peak (CTP)", "s"),
    "mean_contraction_peak_to_min_speed_s": ("Contraction peak to min. speed (CTPMS)", "s"),
    "mean_relaxation_time_to_peak_s": ("Relaxation time-to-peak (RTP)", "s"),
    "mean_relaxation_peak_to_baseline_s": ("Relaxation peak to baseline (RTPB)", "s"),
    "mean_contraction_relaxation_time_s": ("Contraction–relaxation time (CRT)", "s"),
    "mean_time_between_max_speeds_s": ("Time between MCS and MRS", "s"),
    "mean_contraction_relaxation_area": ("Contraction–relaxation area (CRA)", "area"),
    "mean_shortening_area": ("Shortening area (SA)", "area"),
    # Calcium
    "n_transients": ("Number of Ca²⁺ transients", "count"),
    "mean_frequency_per_min": ("Ca²⁺ transient frequency", "per min"),
    "mean_frequency_hz": ("Ca²⁺ transient frequency", "Hz"),
    "mean_inter_peak_interval_s": ("Inter-transient interval", "s"),
    "mean_amplitude_df_f0": ("Ca²⁺ transient amplitude", "ΔF/F₀"),
    "max_amplitude_df_f0": ("Peak Ca²⁺ amplitude", "ΔF/F₀"),
    "mean_start_to_peak_s": ("Start-to-peak time", "s"),
    "mean_peak_to_end_s": ("Peak-to-end time", "s"),
    "mean_transient_duration_s": ("Transient duration", "s"),
    "mean_ctd50_s": ("CTD50", "s"),
    "mean_ctd90_s": ("CTD90", "s"),
    "mean_rise_time_10_90_s": ("Rise time (10–90%)", "s"),
    "mean_decay_tau_s": ("Decay time constant τ", "s"),
    # Morphology
    "n_objects": ("Number of objects", "count"),
    "mean_area_px": ("Mean object area", "px²"),
    "median_area_px": ("Median object area", "px²"),
    "mean_eccentricity": ("Eccentricity", ""),
    "total_covered_area_px": ("Total covered area", "px²"),
    "coverage_fraction": ("Coverage fraction", ""),
    "mean_volume_voxels": ("Mean object volume", "voxels"),
    "median_volume_voxels": ("Median object volume", "voxels"),
    "total_volume_voxels": ("Total volume", "voxels"),
    "alignment_score": ("Alignment score", "0–1"),
    "mean_orientation_deg": ("Mean orientation", "°"),
    "alignment_score_3d": ("Alignment score (3D)", "0–1"),
    "texture_alignment_score": ("Structure-tensor alignment", "0–1"),
    "texture_mean_orientation_deg": ("Structure-tensor orientation", "°"),
    "texture_mean_coherence": ("Coherence", "0–1"),
    "texture_alignment_score_3d": ("Structure-tensor alignment (3D)", "0–1"),
    "texture_mean_fractional_anisotropy": ("Fractional anisotropy", "0–1"),
}

_UNIT_SUFFIXES = {
    "_s": "s",
    "_percent": "%",
    "_hz": "Hz",
    "_px": "px",
    "_deg": "°",
    "_voxels": "voxels",
    "_df_f0": "ΔF/F₀",
    "_per_min": "per min",
}


def _metric_display(key: str, comparison: dict | None = None) -> tuple[str, str]:
    """Clean (title, y-axis label) for a summary field.

    Speed/area units for the optical_flow metrics depend on whether a pixel
    size was given, so they are filled from the videos' own summaries when
    available (comparison["units"]).
    """
    units = (comparison or {}).get("units") or {}
    if key in _METRIC_DISPLAY:
        title, unit = _METRIC_DISPLAY[key]
        if unit == "speed":
            unit = units.get("speed_units", "px/s").replace("um", "µm")
        elif unit == "area":
            unit = units.get("area_units", "px").replace("um", "µm")
        return title, unit
    name = key
    unit = ""
    for suffix, u in _UNIT_SUFFIXES.items():
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            unit = u
            break
    if name.startswith("mean_"):
        name = name[5:]
    name = name.replace("_", " ").strip()
    return (name[:1].upper() + name[1:]) if name else key, unit


# Unit -> y-axis label with the quantity spelled out (a bare "s" reads badly).
_AXIS_LABELS = {
    "s": "Time (s)",
    "BPM": "Beats per minute (BPM)",
    "count": "Count",
    "a.u.": "Amplitude (a.u.)",
    "a.u./s": "Velocity (a.u./s)",
    "Hz": "Frequency (Hz)",
    "per min": "Frequency (min⁻¹)",
    "ΔF/F₀": "ΔF/F₀",
    "px²": "Area (px²)",
    "voxels": "Volume (voxels)",
    "°": "Angle (°)",
    "0–1": "Score (0–1)",
    "µm/s": "Speed (µm/s)",
    "px/s": "Speed (px/s)",
    "µm": "Path length (µm)",
    "px": "Path length (px)",
}


# Journal-style rcParams for the comparison figure only (applied via
# rc_context so the other plots are unaffected). Arial is the usual request
# of biomedical journals; Liberation Sans is its metric-compatible stand-in
# on Linux servers, DejaVu the last fallback.
_PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 7,
    "svg.fonttype": "none",  # keep text editable in Illustrator/Inkscape
    "pdf.fonttype": 42,
}


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


def _resolve_group_colors(comparison: dict, n_groups: int) -> list[str]:
    """Bar colors per group: the user's choice (comparison["group_colors"],
    hex strings in group order, None = default) or the built-in palette."""
    chosen = comparison.get("group_colors") or []
    colors = []
    for i in range(n_groups):
        pick = chosen[i] if i < len(chosen) else None
        if isinstance(pick, str) and len(pick) == 7 and pick.startswith("#"):
            colors.append(pick)
        else:
            colors.append(_GROUP_COLORS[i % len(_GROUP_COLORS)])
    return colors


def _figure_caption(comparison: dict, error_bar: str, p_style: str) -> str:
    """One-line methods caption for the figure footer, in the wording a
    figure legend would use."""
    labels = comparison["labels"]
    n_videos = comparison.get("n_videos") or []
    n_text = ", ".join(f"{lab} n = {n}" for lab, n in zip(labels, n_videos)) if n_videos else ""
    tests = {m["test"] for m in comparison["metrics"]}
    omnibus = " / ".join(_TEST_LABELS.get(t, t) for t in sorted(tests)) or "—"
    posthoc_label = comparison.get("posthoc_label", "post-hoc")
    if len(labels) == 2:
        stats = f"{omnibus}"
        bracket = "Bracket shows the p-value"
    else:
        stats = f"{omnibus}, {posthoc_label}"
        bracket = f"Brackets: each group vs. {labels[0]}"
    if p_style == "stars":
        bracket += "; *p < 0.05, **p < 0.01, ***p < 0.001, ****p < 0.0001, ns not significant"
    elif p_style == "nejm":
        bracket += "; *p < 0.05, **p < 0.01, ***p < 0.001, ns not significant"
    parts = [f"Mean ± {error_bar.upper()}; dots are individual videos", stats, bracket]
    if n_text:
        parts.append(n_text)
    return ". ".join(parts) + "."


def plot_group_comparison(
    comparison: dict, out_path: str, max_metrics: int = 12, panels_dir: str | None = None
) -> list[dict]:
    """Publication-style multi-panel figure, one panel per metric, plus
    (optionally) every panel again as its own figure.

    Combined figure: bar = group mean, thin capped error bar = SEM or SD
    (comparison["error_bar"]), overlaid dots = each video's value, and
    stacked significance lines for each group vs. the first
    (control/reference) group. Panels get letters (A, B, C, ...), a clean
    metric title with the unit on the y-axis, a small gray subtitle with
    the omnibus test p-value, and one footer caption. Saved at 300 dpi and
    as an editable-text SVG next to it.

    panels_dir: when given, each metric is also written as a single-panel
    figure (no letter, no caption; the grouped layout keeps its legend) as
    panel_NN_<metric>.png/.svg for dropping straight into a slide or a
    figure of your own composition. Returns the list of panels written
    ({metric, title, png, svg} with file names).
    """
    metrics = comparison["metrics"][:max_metrics]
    error_bar = comparison.get("error_bar", "sem")
    if error_bar not in ("sem", "sd"):
        error_bar = "sem"
    p_style = comparison.get("p_style", "nejm")
    if p_style not in ("nejm", "value", "stars"):
        p_style = "nejm"

    if not metrics:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No comparable numeric metrics", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return []

    bracket_style = comparison.get("bracket_style", "line")
    if bracket_style not in ("line", "bracket"):
        bracket_style = "line"

    with plt.rc_context(_PUB_RC):
        if comparison.get("layout") == "clustered":
            return _draw_clustered_comparison(
                comparison, error_bar, p_style, bracket_style, out_path, max_metrics, panels_dir
            )
        return _draw_group_comparison(comparison, metrics, error_bar, p_style, out_path, bracket_style, panels_dir)


def _save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=300)
    if path.lower().endswith(".png"):
        fig.savefig(path[:-4] + ".svg")


def _panel_filename(idx: int, metric: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in metric)
    return f"panel_{idx + 1:02d}_{safe}.png"


def _sig_line(ax, x0: float, x1: float, y: float, tick: float, text: str, p_style: str, span: float) -> None:
    """One significance annotation: a flat line (Prism 'line' style) or a
    bracket with end ticks, with the p text centred above it."""
    if tick > 0:
        ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color="black", linewidth=0.7, zorder=6, solid_capstyle="butt")
    else:
        ax.plot([x0, x1], [y, y], color="black", linewidth=0.8, zorder=6, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2,
        y + (0.005 if p_style == "stars" else 0.012) * span,
        text,
        ha="center",
        va="bottom",
        fontsize=8 if p_style == "stars" else 6.5,
    )


def _flat_panel(
    ax, comparison: dict, m: dict, error_bar: str, p_style: str, bracket_style: str, rng, letter: str | None
) -> str:
    """Draw one metric of a flat (one-factor) comparison on ax; returns the title."""
    groups = m["groups"]
    labels = [g["label"] for g in groups]
    n_groups = len(groups)
    xs = np.arange(n_groups)
    # Groups without data for this metric (mean None) get no bar; "n = 0"
    # is written where the bar would be.
    means = [np.nan if g["mean"] is None else float(g["mean"]) for g in groups]
    if error_bar == "sd":
        errs = [0.0 if g.get("std") is None else float(g["std"]) for g in groups]
    else:
        errs = [
            0.0 if g.get("sem") is None and g.get("std") is None
            else float(g.get("sem") if g.get("sem") is not None else (g["std"] / np.sqrt(g["n"]) if g["n"] > 1 else 0.0))
            for g in groups
        ]
    colors = _resolve_group_colors(comparison, n_groups)

    ax.bar(xs, means, width=0.6, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7, zorder=2)
    ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=4)

    all_values: list[float] = []
    for gi, g in enumerate(groups):
        vals = np.asarray(g["values"], dtype=float)
        if len(vals) == 0:
            ax.text(xs[gi], 0, "n = 0", ha="center", va="bottom", fontsize=6, color="#888888")
            continue
        jitter = rng.uniform(-0.14, 0.14, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
        ax.scatter(xs[gi] + jitter, vals, color="white", edgecolor="black", linewidth=0.6, s=14, zorder=5)
        all_values.extend(vals.tolist())

    tops = [mu + e for mu, e in zip(means, errs) if not np.isnan(mu)]
    data_max = max(max(all_values, default=0.0), max(tops, default=0.0))
    data_min = min(min(all_values, default=0.0), 0.0)
    span = (data_max - data_min) or 1.0

    height = data_max + 0.08 * span
    step = 0.11 * span if p_style == "stars" else 0.13 * span
    tick = 0.025 * span if bracket_style == "bracket" else 0.0
    pairs = _bracket_pairs(m, labels)
    for i, j, p in pairs:
        _sig_line(ax, xs[i], xs[j], height, tick, _format_p(p, p_style), p_style, span)
        height += step
    top = height + 0.02 * span if pairs else data_max + 0.12 * span
    ax.set_ylim(data_min - 0.04 * span if data_min < 0 else 0.0, top)

    ax.set_xticks(xs)
    long_labels = n_groups > 3 or max(len(lab) for lab in labels) > 8
    ax.set_xticklabels(labels, rotation=35 if long_labels else 0, ha="right" if long_labels else "center")
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    title, unit = _metric_display(m["metric"], comparison)
    ax.set_ylabel(_AXIS_LABELS.get(unit, unit) if unit else title)
    ax.set_title(title, fontweight="bold", pad=14)
    test_label = _TEST_LABELS.get(m["test"], m["test"])
    subtitle = f"{test_label} {_format_p(m['p_value'])}"
    lmm_pairwise = m.get("lmm_pairwise")
    if lmm_pairwise:
        n_sig_lmm = sum(1 for pw in lmm_pairwise if pw["p_value"] < 0.05)
        subtitle += f"; LMM {n_sig_lmm}/{len(lmm_pairwise)} pairs p < 0.05"
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color="#666666")
    if letter:
        ax.text(-0.28, 1.13, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    return title


def _draw_group_comparison(
    comparison: dict,
    metrics: list,
    error_bar: str,
    p_style: str,
    out_path: str,
    bracket_style: str = "line",
    panels_dir: str | None = None,
) -> list[dict]:
    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    # ~2.3 in per panel: three panels fit a two-column journal figure (~7 in).
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.45 * ncols, 2.55 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    titles = []
    for idx, m in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        titles.append(_flat_panel(ax, comparison, m, error_bar, p_style, bracket_style, rng, letters[idx % 26]))

    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    caption = _figure_caption(comparison, error_bar, p_style)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.6, w_pad=1.2)
    fig.text(0.01, 0.006, caption, ha="left", va="bottom", fontsize=6.5, color="#444444", wrap=True)
    _save_fig(fig, out_path)
    plt.close(fig)

    panels: list[dict] = []
    if panels_dir:
        import os

        for idx, m in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(2.7, 2.8))
            _flat_panel(ax, comparison, m, error_bar, p_style, bracket_style, np.random.default_rng(0), None)
            fig.tight_layout()
            name = _panel_filename(idx, m["metric"])
            _save_fig(fig, os.path.join(panels_dir, name))
            plt.close(fig)
            panels.append({"metric": m["metric"], "title": titles[idx], "png": name, "svg": name[:-4] + ".svg"})
    return panels


def _clustered_panel(
    ax,
    comparison: dict,
    key: str,
    error_bar: str,
    p_style: str,
    bracket_style: str,
    colors: dict,
    rng,
    letter: str | None,
    letter_x: float,
) -> str:
    """Draw one metric of a grouped (two-factor) comparison on ax; returns the title."""
    categories: list[str] = comparison["categories"]
    conditions: list[str] = comparison["conditions"]
    per_cat = {pc["category"]: pc["comparison"] for pc in comparison["per_category"]}
    n_cat, n_cond = len(categories), len(conditions)
    bw = 0.8 / n_cond

    def bar_x(ci: int, ki: int) -> float:
        return ci + (ki - (n_cond - 1) / 2.0) * bw

    all_values: list[float] = []
    tops: list[float] = []
    cluster_max: dict[int, float] = {}
    for ci, cat in enumerate(categories):
        comp = per_cat[cat]
        entry = next((m for m in comp["metrics"] if m["metric"] == key), None)
        if entry is None:
            continue
        for grp in entry["groups"]:
            if grp["label"] not in conditions:
                continue
            ki = conditions.index(grp["label"])
            x = bar_x(ci, ki)
            if grp["mean"] is None:
                ax.text(x, 0, "n = 0", ha="center", va="bottom", fontsize=5.5, color="#888888")
                continue
            err = (grp.get("std") if error_bar == "sd" else grp.get("sem")) or 0.0
            ax.bar(x, grp["mean"], width=bw * 0.92, color=colors[grp["label"]], alpha=0.85, edgecolor="black", linewidth=0.7, zorder=2)
            ax.errorbar(x, grp["mean"], yerr=err, fmt="none", ecolor="black", elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=4)
            vals = np.asarray(grp["values"], dtype=float)
            jitter = rng.uniform(-0.3 * bw, 0.3 * bw, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
            ax.scatter(x + jitter, vals, color="white", edgecolor="black", linewidth=0.5, s=10, zorder=5)
            all_values.extend(vals.tolist())
            top = max(grp["mean"] + err, float(vals.max()) if len(vals) else grp["mean"])
            tops.append(top)
            cluster_max[ci] = max(cluster_max.get(ci, 0.0), top)

    data_max = max(tops, default=1.0)
    data_min = min(min(all_values, default=0.0), 0.0)
    span = (data_max - data_min) or 1.0
    step = 0.11 * span if p_style == "stars" else 0.13 * span
    tick = 0.025 * span if bracket_style == "bracket" else 0.0
    highest = data_max
    for ci, cat in enumerate(categories):
        comp = per_cat[cat]
        entry = next((m for m in comp["metrics"] if m["metric"] == key), None)
        if entry is None or entry.get("test") is None:
            continue
        labels_in_cat = [g["label"] for g in entry["groups"]]
        height = cluster_max.get(ci, data_max) + 0.08 * span
        for i, j, p in _bracket_pairs(entry, labels_in_cat):
            xi = bar_x(ci, conditions.index(labels_in_cat[i]))
            xj = bar_x(ci, conditions.index(labels_in_cat[j]))
            _sig_line(ax, xi, xj, height, tick, _format_p(p, p_style), p_style, span)
            height += step
        highest = max(highest, height)
    ax.set_ylim(data_min - 0.04 * span if data_min < 0 else 0.0, highest + 0.06 * span)

    ax.set_xticks(np.arange(n_cat))
    long_labels = n_cat > 4 or max(len(c) for c in categories) > 8
    ax.set_xticklabels(categories, rotation=35 if long_labels else 0, ha="right" if long_labels else "center")
    ax.set_xlim(-0.6, n_cat - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    title, unit = _metric_display(key, comparison)
    ax.set_ylabel(_AXIS_LABELS.get(unit, unit) if unit else title)
    tw = (comparison.get("two_way_anova") or {}).get(key)
    if tw:
        fa = (comparison.get("factor_names") or {}).get("a") or "category"
        fb = (comparison.get("factor_names") or {}).get("b") or "condition"
        sub = (
            f"Two-way ANOVA: {fa} {_short_p(tw['p_category'])}, {fb} {_short_p(tw['p_condition'])}, "
            f"{fa}×{fb} {_short_p(tw['p_interaction'])}"
        )
        ax.set_title(title, fontweight="bold", pad=14)
        ax.text(0.5, 1.01, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color="#666666")
        if letter:
            ax.text(letter_x, 1.13, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    else:
        ax.set_title(title, fontweight="bold", pad=6)
        if letter:
            ax.text(letter_x, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    return title


def _draw_clustered_comparison(
    comparison: dict,
    error_bar: str,
    p_style: str,
    bracket_style: str,
    out_path: str,
    max_metrics: int,
    panels_dir: str | None = None,
) -> list[dict]:
    """Grouped-bar layout: x clusters = categories (e.g. cell lines), bars
    within a cluster = conditions (e.g. Vehicle / treatment) with one colour
    per condition and a shared legend; significance lines compare the
    conditions within each cluster (vs. the first condition)."""
    import matplotlib.patches as mpatches

    categories: list[str] = comparison["categories"]
    conditions: list[str] = comparison["conditions"]
    metric_keys: list[str] = comparison["metric_keys"][:max_metrics]
    n_cat, n_cond = len(categories), len(conditions)
    cond_colors = comparison.get("condition_colors") or {}
    colors = {
        c: (cond_colors.get(c) if isinstance(cond_colors.get(c), str) else _GROUP_COLORS[(i + 1) % len(_GROUP_COLORS)])
        for i, c in enumerate(conditions)
    }
    rng = np.random.default_rng(0)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    n_bars = n_cat * n_cond
    # Wider clusters get fewer panels per row so the figure stays ~7-8 in wide.
    ncols = 3 if n_bars <= 6 else (2 if n_bars <= 12 else 1)
    ncols = min(ncols, len(metric_keys))
    nrows = int(np.ceil(len(metric_keys) / ncols))
    panel_w = float(np.clip(0.8 + 0.36 * n_bars, 2.45, 7.5))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 2.7 * nrows), squeeze=False)
    letter_x = -0.28 * (2.45 / panel_w)

    titles = []
    for idx, key in enumerate(metric_keys):
        ax = axes[idx // ncols][idx % ncols]
        titles.append(
            _clustered_panel(ax, comparison, key, error_bar, p_style, bracket_style, colors, rng, letters[idx % 26], letter_x)
        )

    for idx in range(len(metric_keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles = [mpatches.Patch(facecolor=colors[c], edgecolor="black", linewidth=0.6, label=c) for c in conditions]
    fig.legend(handles=handles, loc="upper center", ncol=min(n_cond, 6), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.995))

    posthoc_label = comparison.get("posthoc_label", "post-hoc")
    ref = conditions[0]
    parts = [f"Mean ± {error_bar.upper()}; dots are individual videos", f"{posthoc_label} within each category, vs. {ref}"]
    if p_style == "stars":
        parts.append("*p < 0.05, **p < 0.01, ***p < 0.001, ****p < 0.0001, ns not significant")
    elif p_style == "nejm":
        parts.append("*p < 0.05, **p < 0.01, ***p < 0.001, ns not significant")
    n_videos = comparison.get("n_videos") or []
    if n_videos and len(set(n_videos)) == 1:
        parts.append(f"n = {n_videos[0]} videos per bar")
    elif n_videos:
        parts.append(", ".join(f"{lab} n = {n}" for lab, n in zip(comparison["labels"], n_videos)))
    caption = ". ".join(parts) + "."
    fig.tight_layout(rect=(0, 0.035, 1, 0.95), h_pad=1.6, w_pad=1.2)
    fig.text(0.01, 0.006, caption, ha="left", va="bottom", fontsize=6.5, color="#444444", wrap=True)
    _save_fig(fig, out_path)
    plt.close(fig)

    panels: list[dict] = []
    if panels_dir:
        import os

        # Wide enough for the two-way ANOVA subtitle; legend across the top.
        single_w = max(3.9, panel_w + 1.2)
        for idx, key in enumerate(metric_keys):
            fig, ax = plt.subplots(figsize=(single_w, 3.3))
            _clustered_panel(
                ax, comparison, key, error_bar, p_style, bracket_style, colors, np.random.default_rng(0), None, letter_x
            )
            fig.legend(handles=handles, loc="upper center", ncol=min(n_cond, 6), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.995))
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            name = _panel_filename(idx, key)
            _save_fig(fig, os.path.join(panels_dir, name))
            plt.close(fig)
            panels.append({"metric": key, "title": titles[idx], "png": name, "svg": name[:-4] + ".svg"})
    return panels


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
