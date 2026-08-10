"""Two-channel colocalization analysis.

Standard metrics from fluorescence colocalization literature (Manders,
Verschuuren & Bolte 1993/2006; Dunn, Kamocka & McDonald 2011), applied to a
pair of single-channel images (e.g. two fluorescence channels' max-intensity
projections):

- Pearson correlation coefficient (PCC): how linearly related the two
  channels' intensities are, pixel by pixel. Sensitive to both co-
  distribution and to any shared background/bleed-through.
- Manders' overlap coefficient: a non-thresholded measure of co-occurrence,
  in [0, 1] (well-separated signals with no shared background tend toward
  0-ish depending on how much intensity structure they share; two identical
  images give 1).
- Manders' M1 / M2: the more commonly reported pair — the fraction of
  channel A's total intensity that sits in pixels where channel B is above
  its own threshold (M1), and vice versa (M2). Each is in [0, 1] and answers
  a directional question ("of all the A signal, how much overlaps B?"),
  which plain correlation can't.

Per-channel thresholds for Manders M1/M2 use Otsu's method, a simple and
widely available default — NOT the Costes automatic-threshold method some
dedicated colocalization tools use, which is more rigorous but
significantly more involved to implement correctly. Note this choice
explicitly if these numbers go into a publication.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from skimage.filters import threshold_otsu


def analyze_colocalization(channel_a: np.ndarray, channel_b: np.ndarray) -> dict:
    if channel_a.shape != channel_b.shape:
        raise ValueError(
            f"Channel images must be the same shape to compare pixel-by-pixel "
            f"(got {channel_a.shape} and {channel_b.shape})"
        )

    a = channel_a.astype(np.float64)
    b = channel_b.astype(np.float64)
    a_flat, b_flat = a.flatten(), b.flatten()

    pearson_r, pearson_p = stats.pearsonr(a_flat, b_flat)

    denom = np.sqrt(np.sum(a_flat**2) * np.sum(b_flat**2))
    overlap_coefficient = float(np.sum(a_flat * b_flat) / denom) if denom > 0 else None

    threshold_a = float(threshold_otsu(channel_a)) if np.ptp(channel_a) > 0 else float(a.max())
    threshold_b = float(threshold_otsu(channel_b)) if np.ptp(channel_b) > 0 else float(b.max())
    mask_a = channel_a > threshold_a
    mask_b = channel_b > threshold_b

    sum_a = np.sum(a)
    sum_b = np.sum(b)
    manders_m1 = float(np.sum(a[mask_b]) / sum_a) if sum_a > 0 else None
    manders_m2 = float(np.sum(b[mask_a]) / sum_b) if sum_b > 0 else None

    return {
        "n_pixels": int(a.size),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "manders_overlap_coefficient": overlap_coefficient,
        "manders_m1": manders_m1,
        "manders_m2": manders_m2,
        "threshold_a": threshold_a,
        "threshold_b": threshold_b,
        "fraction_a_positive": float(np.mean(mask_a)),
        "fraction_b_positive": float(np.mean(mask_b)),
        "fraction_both_positive": float(np.mean(mask_a & mask_b)),
    }
