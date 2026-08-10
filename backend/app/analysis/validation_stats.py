"""Method-agreement statistics: this tool's values vs. a reference method
(e.g. Fiji/MUSCLEMOTION, manual annotation) on the same recordings.

This is the standard statistical toolkit for validating a new measurement
method against an established one in the biomedical literature:

- Pearson / Spearman correlation quantify how well the two methods track
  each other, but a perfect correlation says nothing about whether the two
  methods actually agree in absolute terms — a constant offset between them
  still correlates perfectly.
- ICC(2,1) (two-way random effects, absolute agreement, single measurement)
  and the Bland-Altman bias / limits of agreement both directly measure
  absolute agreement and will catch a systematic offset that correlation
  misses.

Reference: Bland JM, Altman DG. "Statistical methods for assessing agreement
between two methods of clinical measurement." Lancet. 1986.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater.

    a, b: paired measurements from the two methods, one row per subject.
    Computed via the standard two-way ANOVA mean-square decomposition
    (Shrout & Fleiss, 1979) rather than a stats package, to avoid adding a
    dependency for one formula.
    """
    n = len(a)
    if n < 2:
        return float("nan")
    data = np.stack([a, b], axis=1)  # (n, 2)
    k = 2
    grand_mean = data.mean()
    subject_means = data.mean(axis=1)
    method_means = data.mean(axis=0)

    ss_total = np.sum((data - grand_mean) ** 2)
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_methods = n * np.sum((method_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_methods

    ms_subjects = ss_subjects / (n - 1)
    ms_methods = ss_methods / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0

    denominator = ms_subjects + (k - 1) * ms_error + k * (ms_methods - ms_error) / n
    if denominator == 0:
        return float("nan")
    icc = (ms_subjects - ms_error) / denominator
    return float(icc)


def compute_agreement(values_a: list[float], values_b: list[float]) -> dict:
    """Full method-agreement report for paired (this-tool, reference) values."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("values_a and values_b must be the same length (paired)")
    n = len(a)
    if n < 3:
        raise ValueError("Need at least 3 paired observations for a meaningful agreement analysis")

    pearson_r, pearson_p = stats.pearsonr(a, b)
    spearman_r, spearman_p = stats.spearmanr(a, b)
    icc = _icc_2_1(a, b)

    diffs = a - b
    means = (a + b) / 2.0
    bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff

    # Simple linear regression (b as a function of a) for the scatter plot.
    slope, intercept, r_value, _, _ = stats.linregress(a, b)

    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "icc_2_1": icc,
        "bland_altman_bias": bias,
        "bland_altman_sd_diff": sd_diff,
        "bland_altman_loa_lower": float(loa_lower),
        "bland_altman_loa_upper": float(loa_upper),
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "regression_r_squared": float(r_value**2),
        "values_a": a.tolist(),
        "values_b": b.tolist(),
        "diffs": diffs.tolist(),
        "means": means.tolist(),
    }
