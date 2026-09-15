"""Statistical comparison of a metric across two or more groups of analyzed videos.

Meant for the common "treatment vs control" (or multi-arm dose/condition)
experimental design: run the same single-video analysis (beating / calcium /
morphology) over every video in each group, then compare each numeric
summary metric across groups.

Two test families are available (test_family argument):

- "nonparametric" (default): two groups get a Mann-Whitney U test (rather
  than a t-test) since group sizes in this kind of experiment are usually
  small (a handful of wells/videos per condition) and there's no reason to
  assume normality. Three or more groups get the non-parametric
  equivalent, Kruskal-Wallis, as the omnibus test, followed by Dunn's
  post-hoc test for each pairwise comparison (rank-based, matching
  Kruskal-Wallis's own assumptions) with Bonferroni correction.
- "parametric": the GraphPad-Prism-style workflow used in the lab's own
  manuscripts (Welch's t-test for two groups; one-way ANOVA with Tukey HSD
  post-hoc for 3+, plus Welch's ANOVA and pairwise Welch t-tests with Holm
  correction for the unequal-variance case, reported alongside). Each
  group's Shapiro-Wilk normality p-value is reported so the choice can be
  justified; with n < ~8 per group that test has little power, which is
  why non-parametric stays the default.

When multiple recordings/images come from the same underlying biological
sample (e.g. several fields of view from one differentiation batch/well),
treating each of those as an independent observation is pseudoreplication —
it understates the true variance and inflates the apparent significance,
because repeated measurements of the same sample are correlated with each
other, not independent. Passing a cluster/batch label per video lets
compare_groups additionally fit a linear mixed-effects model (value ~ group
+ (1|cluster)) and report cluster-corrected pairwise p-values for every
group pair (Wald tests via contrasts on a single REML fit), the same
general approach (fixed effect + random intercept for sample identity, fit
by REML, residual normality checked via Shapiro-Wilk) used in Lee et al.,
"IGFBP2 Mediates Human iPSC-Cardiomyocyte Proliferation in a Cellular
Contact-Dependent Manner," Circulation Research, 2025 — generalized here
from their two-group design to arbitrarily many groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.oneway import anova_oneway

TestFamily = Literal["nonparametric", "parametric"]

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


@dataclass
class GroupInput:
    label: str
    summaries: list[dict]
    clusters: list | None = None
    n_videos: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_videos = len(self.summaries)


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


def _dunns_posthoc(labels: list[str], per_group_vals: list[list[float]]) -> list[dict]:
    """Dunn's post-hoc pairwise test following a significant Kruskal-Wallis result.

    Rank-based (matches Kruskal-Wallis's own assumptions, unlike pairwise
    t-tests), with a tie correction on the standard error and Bonferroni
    correction across all pairwise comparisons — the simplest, most
    conservative multiple-comparison correction, chosen over e.g.
    Benjamini-Hochberg FDR for ease of interpretation at the group counts
    this tool is meant for (a handful of conditions).
    """
    all_vals = np.concatenate([np.asarray(v) for v in per_group_vals])
    n_total = len(all_vals)
    ranks = stats.rankdata(all_vals)

    _, tie_counts = np.unique(all_vals, return_counts=True)
    tie_correction = float(np.sum(tie_counts**3 - tie_counts)) / (12.0 * (n_total - 1)) if n_total > 1 else 0.0

    mean_ranks = []
    ns = []
    offset = 0
    for vals in per_group_vals:
        n = len(vals)
        mean_ranks.append(float(ranks[offset : offset + n].mean()))
        ns.append(n)
        offset += n

    n_pairs = len(labels) * (len(labels) - 1) // 2
    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            variance_term = (n_total * (n_total + 1) / 12.0 - tie_correction) * (1.0 / ns[i] + 1.0 / ns[j])
            if variance_term <= 0:
                continue
            se = np.sqrt(variance_term)
            z = (mean_ranks[i] - mean_ranks[j]) / se
            p_raw = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
            p_bonf = min(p_raw * n_pairs, 1.0)
            results.append(
                {
                    "group_a": labels[i],
                    "group_b": labels[j],
                    "z": float(z),
                    "p_value": p_raw,
                    "p_value_bonferroni": p_bonf,
                    # Generic "the corrected p to display" key, shared with
                    # the parametric post-hoc so plots/UI don't branch.
                    "p_adjusted": p_bonf,
                }
            )
    return results


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control, less
    conservative than Bonferroni, no independence assumption)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p_values[idx])))
        adjusted[idx] = running
    return adjusted


def _parametric_posthoc(labels: list[str], per_group_vals: list[list[float]]) -> list[dict]:
    """Every pairwise comparison after a one-way ANOVA.

    Two adjusted p-values per pair: Tukey HSD (assumes equal variances —
    what Prism's default one-way ANOVA workflow reports) and a pairwise
    Welch t-test with Holm correction (no equal-variance assumption — the
    role Dunnett's T3 / Games-Howell play after a Welch/Brown-Forsythe
    ANOVA in Prism; T3 itself isn't in scipy, and Welch + Holm is the
    standard conservative stand-in). `p_adjusted` is Tukey's, matching the
    plain-ANOVA omnibus this tool reports as the primary parametric test.
    """
    try:
        tukey = stats.tukey_hsd(*per_group_vals)
    except Exception:
        tukey = None

    pairs: list[dict] = []
    welch_raw: list[float] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            try:
                p_welch = float(stats.ttest_ind(per_group_vals[i], per_group_vals[j], equal_var=False).pvalue)
            except Exception:
                p_welch = 1.0
            if not np.isfinite(p_welch):
                p_welch = 1.0
            welch_raw.append(p_welch)
            p_tukey = None
            if tukey is not None:
                p_tukey = float(tukey.pvalue[i][j])
                if not np.isfinite(p_tukey):
                    p_tukey = None
            pairs.append(
                {
                    "group_a": labels[i],
                    "group_b": labels[j],
                    "mean_difference": float(np.mean(per_group_vals[j]) - np.mean(per_group_vals[i])),
                    "p_value_tukey": p_tukey,
                    "p_value_welch": p_welch,
                }
            )
    for entry, p_adj in zip(pairs, _holm_adjust(welch_raw)):
        entry["p_value_welch_holm"] = float(p_adj)
        entry["p_adjusted"] = entry["p_value_tukey"] if entry["p_value_tukey"] is not None else entry["p_value_welch_holm"]
    return pairs


def _shapiro_p(values: list[float]) -> float | None:
    if len(values) < 3 or len(set(values)) < 2:
        return None
    try:
        return float(stats.shapiro(values).pvalue)
    except Exception:
        return None


def _fit_lmm_pairwise(
    labels: list[str],
    per_group_vals: list[list[float]],
    per_group_clusters: list[list],
) -> dict | None:
    """Fit value ~ group + (1|cluster) via REML and report every pairwise contrast.

    None if not applicable/fittable — e.g. no cluster labels were given for
    every group, or there's no actual repeated-measurement structure to
    account for (every cluster label is unique, so the random intercept has
    nothing to estimate). Pairwise p-values come from Wald tests on a
    single fit: raw coefficients for pairs involving the reference group
    (labels[0]), and linear contrasts (numeric r_matrix passed to
    statsmodels' t_test) for the rest — this is exact and avoids refitting
    the model once per pair. Contrasts are built as numeric vectors rather
    than the string constraint syntax t_test also accepts, because the
    patsy-generated parameter names here (e.g.
    "C(group, Treatment(reference='W'))[T.X]") contain '=' and quote
    characters that break that string parser.
    """
    if any(len(clus) == 0 or any(c is None for c in clus) for clus in per_group_clusters):
        return None

    n_total = sum(len(v) for v in per_group_vals)
    unique_clusters = {c for clus in per_group_clusters for c in clus}
    if len(unique_clusters) < 2 or len(unique_clusters) >= n_total:
        return None  # no repeated measurements within any cluster

    rows_value: list[float] = []
    rows_group: list[str] = []
    rows_cluster: list[str] = []
    for label, vals, clus in zip(labels, per_group_vals, per_group_clusters):
        rows_value.extend(vals)
        rows_group.extend([label] * len(vals))
        rows_cluster.extend(str(c) for c in clus)
    df = pd.DataFrame({"value": rows_value, "group": rows_group, "cluster": rows_cluster})

    reference = labels[0]
    formula = f"value ~ C(group, Treatment(reference={reference!r}))"
    try:
        model = MixedLM.from_formula(formula, groups="cluster", data=df)
        fit = model.fit(reml=True)
    except Exception:
        return None

    def _coef_name(other_label: str) -> str:
        return f"C(group, Treatment(reference={reference!r}))[T.{other_label}]"

    # r_matrix width must match the fixed-effects vector only (fit.params
    # also includes the random-effect variance component as a trailing
    # entry, which t_test's r_matrix does not cover).
    fe_names = list(fit.fe_params.index)

    def _param_index(other_label: str) -> int | None:
        name = _coef_name(other_label)
        return fe_names.index(name) if name in fe_names else None

    pairwise = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            gi, gj = labels[i], labels[j]
            try:
                if gi == reference:
                    idx = _param_index(gj)
                    if idx is None:
                        continue
                    coefficient = float(fit.fe_params.iloc[idx])
                    p_value = float(fit.pvalues.iloc[idx])
                elif gj == reference:
                    idx = _param_index(gi)
                    if idx is None:
                        continue
                    coefficient = -float(fit.fe_params.iloc[idx])
                    p_value = float(fit.pvalues.iloc[idx])
                else:
                    idx_i, idx_j = _param_index(gi), _param_index(gj)
                    if idx_i is None or idx_j is None:
                        continue
                    r_matrix = np.zeros((1, len(fe_names)))
                    r_matrix[0, idx_j] = 1.0
                    r_matrix[0, idx_i] = -1.0
                    test_result = fit.t_test(r_matrix)
                    coefficient = float(np.asarray(test_result.effect).ravel()[0])
                    p_value = float(np.asarray(test_result.pvalue).ravel()[0])
            except Exception:
                continue
            pairwise.append({"group_a": gi, "group_b": gj, "coefficient": coefficient, "p_value": p_value})

    if not pairwise:
        return None

    residual_shapiro_p = None
    resid = np.asarray(fit.resid)
    if len(resid) >= 3:
        try:
            residual_shapiro_p = float(stats.shapiro(resid).pvalue)
        except Exception:
            residual_shapiro_p = None

    return {
        "lmm_converged": bool(fit.converged),
        "lmm_residual_shapiro_p": residual_shapiro_p,
        "lmm_n_clusters": len(unique_clusters),
        "lmm_pairwise": pairwise,
    }


def compare_groups(groups: list[GroupInput], test_family: TestFamily = "nonparametric") -> dict:
    """Compare every shared numeric metric across two or more groups of per-video summaries.

    test_family selects the primary test (see module docstring): rank-based
    Mann-Whitney U / Kruskal-Wallis + Dunn's ("nonparametric", default), or
    Welch's t-test / one-way ANOVA + Tukey HSD ("parametric"). Every
    post-hoc entry carries a `p_adjusted` key regardless of family.

    Each group's `clusters` (optional): a batch/sample label per entry in
    that group's summaries, same length and order. When every group has
    cluster labels, each metric additionally gets cluster-aware linear
    mixed-model pairwise p-values alongside the primary test, correcting
    for repeated measurements sharing a cluster label.
    """
    if len(groups) < 2:
        raise ValueError("compare_groups needs at least 2 groups")
    if test_family not in ("nonparametric", "parametric"):
        raise ValueError("test_family must be 'nonparametric' or 'parametric'")

    keys = set()
    for g in groups:
        for s in g.summaries:
            for k, v in s.items():
                if k in _EXCLUDED_METRICS:
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                keys.add(k)

    labels = [g.label for g in groups]
    metrics = []
    for key in sorted(keys):
        per_group_vals: list[list[float]] = []
        per_group_clusters: list[list] = []
        for g in groups:
            vals, clus = _numeric_values_with_clusters(g.summaries, key, g.clusters)
            per_group_vals.append(vals)
            per_group_clusters.append(clus)
        if any(len(v) == 0 for v in per_group_vals):
            continue

        entry: dict = {
            "metric": key,
            "groups": [
                {
                    "label": label,
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                    "shapiro_p": _shapiro_p(vals),
                    "values": vals,
                }
                for label, vals in zip(labels, per_group_vals)
            ],
            "statistic": None,
            "p_value": None,
        }

        has_variance = len({v for vals in per_group_vals for v in vals}) > 1

        if len(groups) == 2:
            if test_family == "nonparametric":
                entry["test"] = "mann_whitney_u"
                if has_variance:
                    try:
                        result = stats.mannwhitneyu(per_group_vals[0], per_group_vals[1], alternative="two-sided")
                        entry["statistic"] = float(result.statistic)
                        entry["p_value"] = float(result.pvalue)
                    except ValueError:
                        pass
            else:
                entry["test"] = "welch_t"
                if has_variance:
                    try:
                        result = stats.ttest_ind(per_group_vals[0], per_group_vals[1], equal_var=False)
                        entry["statistic"] = float(result.statistic)
                        entry["p_value"] = float(result.pvalue) if np.isfinite(result.pvalue) else None
                        student = stats.ttest_ind(per_group_vals[0], per_group_vals[1], equal_var=True)
                        entry["student_t_p_value"] = float(student.pvalue) if np.isfinite(student.pvalue) else None
                    except ValueError:
                        pass
        else:
            if test_family == "nonparametric":
                entry["test"] = "kruskal_wallis"
                if has_variance:
                    try:
                        result = stats.kruskal(*per_group_vals)
                        entry["statistic"] = float(result.statistic)
                        entry["p_value"] = float(result.pvalue)
                    except ValueError:
                        pass
                entry["posthoc"] = _dunns_posthoc(labels, per_group_vals) if entry["p_value"] is not None else None
            else:
                entry["test"] = "anova"
                entry["welch_anova_p_value"] = None
                if has_variance:
                    try:
                        result = stats.f_oneway(*per_group_vals)
                        entry["statistic"] = float(result.statistic)
                        entry["p_value"] = float(result.pvalue) if np.isfinite(result.pvalue) else None
                    except ValueError:
                        pass
                    try:
                        welch = anova_oneway(per_group_vals, use_var="unequal", welch_correction=True)
                        entry["welch_anova_p_value"] = float(welch.pvalue) if np.isfinite(welch.pvalue) else None
                    except Exception:
                        pass
                entry["posthoc"] = _parametric_posthoc(labels, per_group_vals) if entry["p_value"] is not None else None

        lmm = _fit_lmm_pairwise(labels, per_group_vals, per_group_clusters)
        if lmm is not None:
            entry.update(lmm)

        metrics.append(entry)

    # Most-significant-first (by the primary omnibus p-value) so the
    # interesting differences surface immediately.
    metrics.sort(key=lambda m: (m["p_value"] is None, m["p_value"] if m["p_value"] is not None else 0.0))

    if len(groups) == 2:
        posthoc_label = "Mann-Whitney U" if test_family == "nonparametric" else "Welch's t-test"
    else:
        posthoc_label = (
            "Dunn's post-hoc (Bonferroni)" if test_family == "nonparametric" else "Tukey HSD post-hoc"
        )

    return {
        "labels": labels,
        "n_videos": [g.n_videos for g in groups],
        "test_family": test_family,
        "posthoc_label": posthoc_label,
        "metrics": metrics,
    }
