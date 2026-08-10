"""Shared population-alignment (order parameter) math.

Used both by per-object orientation (morphology.py, one angle/axis per
segmented cell) and per-pixel/voxel orientation (structure_tensor.py, one
angle/axis per pixel with an optional reliability weight). Both boil down to
the same circular (2D) / nematic (3D) order-parameter construction, just with
or without weights, so the logic lives in one place.
"""
from __future__ import annotations

import numpy as np


def circular_alignment_2d(
    orientations_rad: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float | None, float | None]:
    """Population alignment score for axial (mod-pi) 2D orientation angles.

    Doubling the angle maps the axial [-pi/2, pi/2] range onto a full circle
    so the usual circular mean / resultant length works (an object pointing
    at +85 deg and one at -85 deg are nearly parallel, not opposite).
    Returns (alignment_score in [0,1], mean_orientation_deg); 1 = perfectly
    aligned, 0 = uniformly random orientation.

    weights (e.g. structure-tensor coherence) down-weight unreliable/
    near-isotropic samples instead of letting them dilute the average with
    an arbitrary angle. Defaults to uniform weighting.
    """
    n = len(orientations_rad)
    if n < 2:
        return None, None
    if weights is None:
        weights = np.ones(n)
    total_weight = np.sum(weights)
    if total_weight <= 0:
        return None, None
    resultant = np.sum(weights * np.exp(2j * orientations_rad)) / total_weight
    alignment_score = float(np.abs(resultant))
    mean_orientation_deg = float(np.degrees(0.5 * np.angle(resultant)))
    return alignment_score, mean_orientation_deg


def nematic_alignment_3d(
    directions: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float | None, np.ndarray | None]:
    """Population alignment score for 3D axes (headless unit vectors).

    Standard nematic order-parameter construction (liquid-crystal / fiber
    alignment literature): Q = <3 n(x)n - I> / 2 is invariant to n -> -n, so
    it handles the sign ambiguity of a "long axis" correctly. Its largest
    eigenvalue S in [0,1] is the alignment score (0 = isotropic/random,
    1 = perfectly aligned) and the matching eigenvector is the mean
    alignment direction.

    weights (e.g. structure-tensor fractional anisotropy) down-weight
    unreliable/near-isotropic samples. Defaults to uniform weighting.
    """
    n = len(directions)
    if n < 2:
        return None, None
    if weights is None:
        weights = np.ones(n)
    total_weight = np.sum(weights)
    if total_weight <= 0:
        return None, None
    q_tensors = np.array([w * (3.0 * np.outer(d, d) - np.eye(3)) for d, w in zip(directions, weights)])
    q_mean = np.sum(q_tensors, axis=0) / (2.0 * total_weight)
    eigvals, eigvecs = np.linalg.eigh(q_mean)
    order = np.argsort(eigvals)[::-1]
    s = float(max(eigvals[order[0]], 0.0))
    director = eigvecs[:, order[0]]
    return s, director
