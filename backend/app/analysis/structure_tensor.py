"""Structure-tensor-based local orientation analysis (2D and 3D).

This is the same technique used by Fiji's OrientationJ plugin and by
Cardiotensor for cardiac fiber/helical-angle mapping: instead of segmenting
individual objects and averaging their long axes (see morphology.py), the
local dominant orientation is estimated directly from the image gradient at
every pixel/voxel. That makes it applicable to textures that aren't made of
discrete, segmentable objects — striated sarcomere pattern, fibrous ECM,
tissue-scale fiber orientation — and it doesn't depend on a threshold/
watershed segmentation being correct.

Method: smooth the image at a "noise scale" sigma, take its gradient, form
the outer product of the gradient with itself at every pixel, then smooth
that (the "structure tensor") at a coarser "integration scale" rho. The
eigenvector of the tensor's smallest eigenvalue points along the local
structure's long axis (intensity varies *least* along the direction a fiber
or edge runs), and the eigenvalue spread gives a coherence/anisotropy score
for how reliable that direction estimate is (0 in a flat/isotropic region,
approaching 1 along a sharp, consistently-oriented edge or fiber).

Reference for the coherence/anisotropy formulas: Rezakhaniha et al.,
"Experimental investigation of collagen waviness and orientation in the
arterial adventitia using confocal laser scanning microscopy," Biomech
Model Mechanobiol, 2012 (the method behind Fiji's OrientationJ).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from app.analysis.alignment_stats import circular_alignment_2d, nematic_alignment_3d


def _gaussian_derivative(image: np.ndarray, axis: int, sigma: float) -> np.ndarray:
    order = [0] * image.ndim
    order[axis] = 1
    return ndi.gaussian_filter(image, sigma=sigma, order=order)


def compute_structure_tensor_2d(
    image: np.ndarray,
    noise_sigma: float = 1.0,
    integration_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel local orientation (radians, axial [-pi/2, pi/2]) and coherence [0,1].

    image: (H, W) grayscale array.
    """
    img = image.astype(np.float64)
    iy = _gaussian_derivative(img, axis=0, sigma=noise_sigma)
    ix = _gaussian_derivative(img, axis=1, sigma=noise_sigma)

    jxx = ndi.gaussian_filter(ix * ix, sigma=integration_sigma)
    jyy = ndi.gaussian_filter(iy * iy, sigma=integration_sigma)
    jxy = ndi.gaussian_filter(ix * iy, sigma=integration_sigma)

    # 0.5*arctan2(2*Jxy, Jxx-Jyy) gives the angle of the *larger*-eigenvalue
    # eigenvector of J, i.e. the gradient direction (steepest intensity
    # change, across a fiber/edge). Structure orientation conventionally
    # means the fiber/edge direction itself — the *smaller*-eigenvalue
    # eigenvector, perpendicular to the gradient — so rotate by 90 degrees
    # and wrap back into the axial (-pi/2, pi/2] range.
    gradient_direction = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    orientation = np.mod(gradient_direction + np.pi / 2 + np.pi / 2, np.pi) - np.pi / 2

    trace = jxx + jyy
    coherence = np.zeros_like(trace)
    valid = trace > 1e-12
    coherence[valid] = np.sqrt((jxx[valid] - jyy[valid]) ** 2 + 4 * jxy[valid] ** 2) / trace[valid]

    return orientation, coherence


def compute_structure_tensor_3d(
    volume: np.ndarray,
    noise_sigma: float = 1.0,
    integration_sigma: float = 2.0,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel local principal direction and fractional anisotropy [0,1].

    volume: (Z, H, W) grayscale array.
    stride subsamples the (already smoothed) tensor field before the
    per-voxel eigendecomposition, which is the expensive step — full
    resolution is rarely needed for a population alignment estimate and can
    be very slow/memory-heavy on large stacks. Returns arrays downsampled by
    `stride` in every axis: principal direction (Z', H', W', 3) in (z,y,x)
    order, and fractional anisotropy (Z', H', W').
    """
    vol = volume.astype(np.float64)
    iz = _gaussian_derivative(vol, axis=0, sigma=noise_sigma)
    iy = _gaussian_derivative(vol, axis=1, sigma=noise_sigma)
    ix = _gaussian_derivative(vol, axis=2, sigma=noise_sigma)

    def smooth(a: np.ndarray) -> np.ndarray:
        return ndi.gaussian_filter(a, sigma=integration_sigma)

    jzz, jyy, jxx = smooth(iz * iz), smooth(iy * iy), smooth(ix * ix)
    jzy, jzx, jyx = smooth(iz * iy), smooth(iz * ix), smooth(iy * ix)

    sl = (slice(None, None, stride),) * 3
    jzz, jyy, jxx = jzz[sl], jyy[sl], jxx[sl]
    jzy, jzx, jyx = jzy[sl], jzx[sl], jyx[sl]

    shape = jzz.shape
    tensor = np.zeros((*shape, 3, 3))
    tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2] = jzz, jyy, jxx
    tensor[..., 0, 1] = tensor[..., 1, 0] = jzy
    tensor[..., 0, 2] = tensor[..., 2, 0] = jzx
    tensor[..., 1, 2] = tensor[..., 2, 1] = jyx

    eigvals, eigvecs = np.linalg.eigh(tensor)  # ascending eigenvalues
    principal = eigvecs[..., :, 0]  # smallest-eigenvalue eigenvector = fiber direction

    l1, l2, l3 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]
    mean_l = (l1 + l2 + l3) / 3.0
    denom = np.sqrt(l1**2 + l2**2 + l3**2)
    fa = np.zeros_like(mean_l)
    valid = denom > 1e-12
    fa[valid] = np.sqrt(1.5) * np.sqrt(
        (l1[valid] - mean_l[valid]) ** 2 + (l2[valid] - mean_l[valid]) ** 2 + (l3[valid] - mean_l[valid]) ** 2
    ) / denom[valid]

    return principal, fa


def structure_tensor_alignment_2d(
    image: np.ndarray,
    noise_sigma: float = 1.0,
    integration_sigma: float = 3.0,
) -> dict:
    """Population alignment summary for a 2D image, plus the maps for plotting."""
    orientation, coherence = compute_structure_tensor_2d(image, noise_sigma, integration_sigma)
    alignment_score, mean_orientation_deg = circular_alignment_2d(
        orientation.flatten(), weights=coherence.flatten()
    )
    return {
        "alignment_score": alignment_score,
        "mean_orientation_deg": mean_orientation_deg,
        "mean_coherence": float(np.mean(coherence)),
        "orientation_map": orientation,
        "coherence_map": coherence,
    }


def structure_tensor_alignment_3d(
    volume: np.ndarray,
    noise_sigma: float = 1.0,
    integration_sigma: float = 2.0,
    stride: int = 2,
) -> dict:
    """Population alignment summary for a 3D volume, plus the maps for plotting."""
    principal, fa = compute_structure_tensor_3d(volume, noise_sigma, integration_sigma, stride)
    alignment_score, director = nematic_alignment_3d(principal.reshape(-1, 3), weights=fa.flatten())
    return {
        "alignment_score_3d": alignment_score,
        "mean_direction_zyx": [float(v) for v in director] if director is not None else None,
        "mean_fractional_anisotropy": float(np.mean(fa)),
        "principal_direction_map": principal,
        "fa_map": fa,
    }
