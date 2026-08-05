"""Statistical comparison of a metric across two groups of analyzed videos.

Meant for the common "treatment vs control" experimental design: run the
same single-video analysis (beating / calcium / morphology) over every
video in each group, then compare each numeric summary metric between
groups with a two-sample test.

Mann-Whitney U (rather than a t-test) is used by default since group sizes
in this kind of experiment are usually small (a handful of wells/videos per
condition) and there's no reason to assume normality.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

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


def _numeric_values(summaries: list[dict], key: str) -> list[float]:
    values = []
    for s in summaries:
        v = s.get(key)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)) and np.isfinite(v):
            values.append(float(v))
    return values


def compare_groups(
    summaries_a: list[dict],
    summaries_b: list[dict],
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> dict:
    """Compare every shared numeric metric between two groups of per-video summaries."""
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
        vals_a = _numeric_values(summaries_a, key)
        vals_b = _numeric_values(summaries_b, key)
        if len(vals_a) == 0 or len(vals_b) == 0:
            continue

        p_value = None
        statistic = None
        if len(vals_a) >= 1 and len(vals_b) >= 1 and (len(set(vals_a)) > 1 or len(set(vals_b)) > 1):
            try:
                result = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            except ValueError:
                pass

        metrics.append(
            {
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
        )

    # Most-significant-first so the interesting differences surface immediately.
    metrics.sort(key=lambda m: (m["p_value"] is None, m["p_value"] if m["p_value"] is not None else 0.0))

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_videos_a": len(summaries_a),
        "n_videos_b": len(summaries_b),
        "metrics": metrics,
    }
