"""FFT-based windowed cross-correlation particle image velocimetry (PIV).

Same algorithm family as PIVlab's piv_FFTmulti — the function used by
PIV-MyoMonitor (github.com/soahleelab/PIV-MyoMonitor), a MATLAB tool built
for cardiac organoid contractility analysis: frames are divided into
interrogation windows, and for each window the displacement between two
frames is found as the peak of the FFT-based cross-correlation, refined to
sub-pixel accuracy via 3-point Gaussian peak interpolation.

This is a *single-pass* implementation (one fixed window size, no
iterative window deformation / adaptive multi-pass refinement the way
PIVlab does it), which is simpler and faster but less accurate for large
displacements or fine spatial structure. Treat it as "the same class of
algorithm," not a numerical drop-in replacement for PIVlab — if bit-for-bit
equivalence to a PIVlab-based result matters, validate against PIVlab
directly (see the app's /api/validate/agreement endpoint) rather than
assuming it here.

Unlike the reference PIV-MyoMonitor tool, this runs fully automatically —
no manual ROI selection, scale-bar measurement, or interactive peak
curation — trading some of PIVlab's accuracy for batch-friendliness.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Empirically calibrated on two otherwise-identical synthetic test videos:
# a flat-background pulsing disk (median per-window intensity std ~0.5,
# PIV mode produced garbage — 15 spurious "beats" vs the true 6) and the
# same video with synthetic speckle noise added (median ~11.5, PIV mode
# correctly detected all 6 beats at the true BPM). The threshold sits
# roughly midway on a log scale between those two measurements.
LOW_TEXTURE_MEDIAN_STD_THRESHOLD = 3.0


def assess_texture(frame: np.ndarray, window_size: int = 32, step: int | None = None) -> dict:
    """Check whether a frame has enough local intensity variation for PIV.

    FFT cross-correlation needs correlatable texture/speckle inside each
    interrogation window: a flat or smoothly-shaded region (e.g. a plain
    bright-field cell body against a plain background, with no visible
    granularity) has a broad, ambiguous, or near-zero correlation peak, so
    the displacement estimate there is dominated by noise rather than real
    motion. This computes the standard deviation of pixel intensities
    within each interrogation window on a single frame and returns the
    median across all windows, plus whether it falls below an empirically
    calibrated low-texture threshold.
    """
    if step is None:
        step = window_size // 2
    h, w = frame.shape
    a = frame.astype(np.float64)
    stds = [
        float(a[y : y + window_size, x : x + window_size].std())
        for y in range(0, h - window_size + 1, step)
        for x in range(0, w - window_size + 1, step)
    ]
    median_std = float(np.median(stds)) if stds else 0.0
    return {
        "median_window_std": median_std,
        "low_texture": median_std < LOW_TEXTURE_MEDIAN_STD_THRESHOLD,
    }


def _gaussian_subpixel(corr: np.ndarray, py: int, px: int) -> tuple[float, float]:
    """3-point Gaussian sub-pixel refinement around a discrete correlation peak.

    Standard PIV sub-pixel estimator: fits a Gaussian through the peak and
    its immediate neighbors along each axis independently. Falls back to 0
    (keep the integer-pixel peak) wherever the neighbors aren't usable
    (peak on the window border, or non-positive correlation values, which
    would make the log undefined).
    """
    h, w = corr.shape
    dy = dx = 0.0
    if 0 < py < h - 1:
        c_m1, c_0, c_p1 = corr[py - 1, px], corr[py, px], corr[py + 1, px]
        if c_m1 > 0 and c_0 > 0 and c_p1 > 0:
            denom = np.log(c_m1) - 2 * np.log(c_0) + np.log(c_p1)
            if abs(denom) > 1e-12:
                dy = 0.5 * (np.log(c_m1) - np.log(c_p1)) / denom
    if 0 < px < w - 1:
        c_m1, c_0, c_p1 = corr[py, px - 1], corr[py, px], corr[py, px + 1]
        if c_m1 > 0 and c_0 > 0 and c_p1 > 0:
            denom = np.log(c_m1) - 2 * np.log(c_0) + np.log(c_p1)
            if abs(denom) > 1e-12:
                dx = 0.5 * (np.log(c_m1) - np.log(c_p1)) / denom
    return dy, dx


def compute_piv_field(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    window_size: int = 32,
    step: int | None = None,
) -> dict:
    """Displacement vector field from frame_a to frame_b via FFT cross-correlation.

    Returns dict with x, y (interrogation window center coordinates, 2D
    grids) and u, v (displacement in x/y, pixels — positive u = rightward,
    positive v = downward, array/image convention), all shaped
    (n_rows, n_cols).

    All windows are cross-correlated in one batched FFT call (numpy's fft2
    treats leading array axes as a batch dimension) rather than one Python
    loop iteration + FFT call per window. Numerically identical to looping
    window-by-window, just much faster — this was previously the dominant
    cost for real (not small-synthetic-test-sized) video: a 500x500 frame
    at the default window_size=32/step=16 has ~900 windows, so per-frame-
    pair cost was ~900 separate small FFT calls with Python-loop overhead
    on top.
    """
    if frame_a.shape != frame_b.shape:
        raise ValueError(f"Frame shapes must match (got {frame_a.shape} and {frame_b.shape})")
    if step is None:
        step = window_size // 2
    h, w = frame_a.shape
    ys = list(range(0, h - window_size + 1, step))
    xs = list(range(0, w - window_size + 1, step))
    if not ys or not xs:
        raise ValueError(f"window_size ({window_size}) is larger than the frame ({h}x{w})")

    a = frame_a.astype(np.float64)
    b = frame_b.astype(np.float64)

    # sliding_window_view is a zero-copy view over every possible window
    # position; np.ix_ then pulls out just the strided (step-spaced) grid
    # of windows we actually want, as one small copy.
    windows_a = sliding_window_view(a, (window_size, window_size))[np.ix_(ys, xs)]
    windows_b = sliding_window_view(b, (window_size, window_size))[np.ix_(ys, xs)]
    windows_a = windows_a - windows_a.mean(axis=(-2, -1), keepdims=True)
    windows_b = windows_b - windows_b.mean(axis=(-2, -1), keepdims=True)

    fa = np.fft.fft2(windows_a)
    fb = np.fft.fft2(windows_b)
    # conj(fa)*fb (not fa*conj(fb)) so the correlation peak lands at
    # +shift when frame_b's content is frame_a's shifted by +shift —
    # i.e. u/v come out as "how frame_b moved relative to frame_a,"
    # verified against scipy.ndimage.shift with a known offset.
    corr = np.fft.fftshift(np.fft.ifft2(np.conj(fa) * fb).real, axes=(-2, -1))

    n_rows, n_cols = len(ys), len(xs)
    flat_idx = np.argmax(corr.reshape(n_rows, n_cols, -1), axis=-1)
    py_grid, px_grid = np.unravel_index(flat_idx, (window_size, window_size))

    x_grid, y_grid = np.meshgrid(
        [x + window_size / 2 for x in xs], [y + window_size / 2 for y in ys]
    )
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))
    center = window_size // 2

    # Only the cheap, non-FFT sub-pixel refinement step still loops per
    # window (it just reads a few neighboring correlation values).
    for i in range(n_rows):
        for j in range(n_cols):
            py, px = int(py_grid[i, j]), int(px_grid[i, j])
            # Sub-pixel Gaussian fit needs strictly positive correlation
            # values (it works in log-space); shifting doesn't change where
            # the peak is, only what the fit sees around it.
            corr_shifted = corr[i, j] - corr[i, j].min() + 1e-6
            dy_sub, dx_sub = _gaussian_subpixel(corr_shifted, py, px)
            v[i, j] = (py - center) + dy_sub
            u[i, j] = (px - center) + dx_sub

    return {"x": x_grid, "y": y_grid, "u": u, "v": v}


def compute_piv_motion_signal(
    frames: np.ndarray,
    window_size: int = 32,
    step: int | None = None,
) -> np.ndarray:
    """Per-frame-transition scalar motion signal for beat detection.

    frames: (N, H, W) grayscale array. Returns an array of length N-1 (mean
    displacement-vector magnitude across all interrogation windows, one
    value per consecutive frame pair) — a richer, vector-field-derived
    analogue of the plain pixel-difference signal used by the default
    beating-analysis mode.
    """
    n = frames.shape[0]
    signal = np.zeros(max(n - 1, 0))
    for i in range(n - 1):
        field = compute_piv_field(frames[i], frames[i + 1], window_size=window_size, step=step)
        magnitude = np.sqrt(field["u"] ** 2 + field["v"] ** 2)
        signal[i] = float(np.mean(magnitude))
    return signal
