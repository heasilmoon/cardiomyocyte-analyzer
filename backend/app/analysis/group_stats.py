"""Statistical comparison of a metric across two groups of analyzed videos.

Meant for the common "treatment vs control" experimental design: run the
same single-video analysis (beating / calcium / morphology) over every
video in each group, then compare each numeric summary metric between
groups with a two-sample test.

Mann-Whitney U (rather than a t-test) is used by default since group sizes
in this kind of experiment are usually small (a handful of wells/videos per
condition) and there's no reason to assume normality.

When multiple recordings/images come from the same underlying biological
sample (e.g. several fields of view from one differentiation batch/well),
treating each of those as an independent observation is pseudoreplication —
it understates the true variance and inflates the apparent significance,
because repeated measurements of the same sample are correlated with each
other, not independent. Passing a cluster/batch label per video lets
compare_groups additionally fit a linear mixed-effects model (value ~ group
+ (1|cluster)) that accounts for this, the same general approach (fixed
effect + random intercept for sample identity, fit by REML, residual
normality checked via Shapiro-Wilk) used in Lee et al., "IGFBP2 Mediates
Human iPSC-Cardiomyocyte Proliferation in a Cellular Contact-Dependent
Manner," Circulation Research, 2025.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

# Per-video technical/bookkeeping fields, not biological measurements —
# comparing them across groups isn't meaningful, so they're left out of the
# statistical comparison even though they're numeric.
_EXCLUDED_METRICS = {
    "reference_frame_index",
    "n_frames",
    "image_area_px",
    "duration_s",
    # Beating auto-tuning diagnostics (see analyze_beating): these are
    # derived directly from the estimated beat period, so they're redundant
    # with mean_bpm / mean_inter_beat_interval_s and would otherwise show
    # up as extra "significant" hits that aren't independent findings.
    "estimated_period_s",
    "min_bpm_gap_used",
    "smoothing_window_s",
}


def _numeric_values_with_clusters(
    summaries: list[dict],
    key: str,
    clusters: list | None,
) -> tuple[list[float], list]:
    """Numeric values for `key`, keeping any parallel cluster labels in lockstep.

    Rows with a missing/non-numeric value for this metric are dropped from
    both lists together, so values[i] and clusters[i] always refer to the
    same source row.
    """
    values: list[float] = []
    kept_clusters: list = []
    for i, s in enumerate(summaries):
        v = s.get(key)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)) and np.isfinite(v):
            values.append(float(v))
            kept_clusters.append(clusters[i] if clusters is not None else None)
    return values, kept_clusters


def _fit_lmm(
    vals_a: list[float],
    clusters_a: list,
    vals_b: list[float],
    clusters_b: list,
) -> dict | None:
    """Fit value ~ group + (1|cluster) via REML; None if not applicable/fittable.

    Only attempted when there's actual repeated-measurement structure to
    account for (at least one cluster appears more than once) — otherwise
    the random intercept has nothing to estimate and the fit is either
    degenerate or offers no benefit over the simpler group-comparison test.
    """
    if not clusters_a or not clusters_b or any(c is None for c in clusters_a + clusters_b):
        return None

    n_total = len(vals_a) + len(vals_b)
    unique_clusters = set(clusters_a) | set(clusters_b)
    if len(unique_clusters) < 2 or len(unique_clusters) >= n_total:
        return None  # no repeated measurements within any cluster

    df = pd.DataFrame(
        {
            "value": vals_a + vals_b,
            "group": (["A"] * len(vals_a)) + (["B"] * len(vals_b)),
            "cluster": [str(c) for c in clusters_a] + [str(c) for c in clusters_b],
        }
    )

    try:
        model = MixedLM.from_formula("value ~ group", groups="cluster", data=df)
        fit = model.fit(reml=True)
    except Exception:
        return None

    coef_name = "group[T.B]"
    if coef_name not in fit.params:
        return None

    residual_shapiro_p = None
    resid = np.asarray(fit.resid)
    if len(resid) >= 3:
        try:
            residual_shapiro_p = float(stats.shapiro(resid).pvalue)
        except Exception:
            residual_shapiro_p = None

    return {
        "lmm_coefficient": float(fit.params[coef_name]),
        "lmm_p_value": float(fit.pvalues[coef_name]),
        "lmm_converged": bool(fit.converged),
        "lmm_residual_shapiro_p": residual_shapiro_p,
        "lmm_n_clusters": len(unique_clusters),
    }


def compare_groups(
    summaries_a: list[dict],
    summaries_b: list[dict],
    label_a: str = "Group A",
    label_b: str = "Group B",
    clusters_a: list | None = None,
    clusters_b: list | None = None,
) -> dict:
    """Compare every shared numeric metric between two groups of per-video summaries.

    clusters_a / clusters_b (optional): a batch/sample label per entry in
    summaries_a / summaries_b, same length and order. When given, each
    metric additionally gets a linear-mixed-model p-value alongside the
    default Mann-Whitney U, correcting for repeated measurements sharing a
    cluster label. Mann-Whitney U is still always computed and remains the
    primary `p_value` field, so behavior without cluster labels is
    unchanged.
    """
    keys = set()
    for s in summaries_a + summaries_b:
        for k, v in s.items():
            if k in _EXCLUDED_METRICS:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            keys.add(k)

    metrics = []
    for key in sorted(keys):
        vals_a, clus_a = _numeric_values_with_clusters(summaries_a, key, clusters_a)
        vals_b, clus_b = _numeric_values_with_clusters(summaries_b, key, clusters_b)
        if len(vals_a) == 0 or len(vals_b) == 0:
            continue

        p_value = None
        statistic = None
        if len(set(vals_a)) > 1 or len(set(vals_b)) > 1:
            try:
                result = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            except ValueError:
                pass

        entry = {
            "metric": key,
            "n_a": len(vals_a),
            "mean_a": float(np.mean(vals_a)),
            "std_a": float(np.std(vals_a, ddof=1)) if len(vals_a) > 1 else 0.0,
            "values_a": vals_a,
            "n_b": len(vals_b),
            "mean_b": float(np.mean(vals_b)),
            "std_b": float(np.std(vals_b, ddof=1)) if len(vals_b) > 1 else 0.0,
            "values_b": vals_b,
            "statistic": statistic,
            "p_value": p_value,
        }

        lmm = _fit_lmm(vals_a, clus_a, vals_b, clus_b)
        if lmm is not None:
            entry.update(lmm)

        metrics.append(entry)

    # Most-significant-first (by the primary Mann-Whitney U p-value) so the
    # interesting differences surface immediately.
    metrics.sort(key=lambda m: (m["p_value"] is None, m["p_value"] if m["p_value"] is not None else 0.0))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_videos_a": len(summaries_a),
        "n_videos_b": len(summaries_b),
        "metrics": metrics,
    }
