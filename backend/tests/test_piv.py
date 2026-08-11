import numpy as np
from scipy import ndimage as ndi

from app.analysis.piv import (
    assess_texture,
    compute_piv_field,
    compute_piv_motion_signal,
    compute_window_texture_mask,
)


def _speckle_image(size=200, seed=0, sigma=1.0):
    """Synthetic PIV tracer-like texture: fine, sharply-localized speckle.

    PIV correlation accuracy depends on having well-localized (not overly
    smooth/blurry) texture to correlate — this mimics that, roughly like a
    seeded/textured microscopy frame rather than a flat gradient.
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size, size)).astype(np.float64)
    return ndi.gaussian_filter(img, sigma=sigma)


def test_piv_recovers_known_subpixel_shift():
    frame_a = _speckle_image()
    for true_dx, true_dy in [(2.0, 0.0), (0.0, 3.0), (1.5, -2.3), (-3.0, 1.0)]:
        frame_b = ndi.shift(frame_a, shift=(true_dy, true_dx), mode="reflect", order=3)
        field = compute_piv_field(frame_a, frame_b, window_size=32, step=16)
        # Skip the border row/col of windows: near the frame edge, ndi.shift's
        # 'reflect' boundary handling and the interrogation window itself
        # both lose valid overlapping content, which is a real, well-known
        # PIV edge effect — not what this test is checking.
        u_interior = field["u"][2:-2, 2:-2]
        v_interior = field["v"][2:-2, 2:-2]
        assert abs(u_interior.mean() - true_dx) < 0.15
        assert abs(v_interior.mean() - true_dy) < 0.15


def test_piv_field_shape_matches_window_grid():
    frame_a = _speckle_image()
    frame_b = frame_a.copy()
    field = compute_piv_field(frame_a, frame_b, window_size=32, step=16)
    h, w = frame_a.shape
    expected_windows = len(range(0, h - 32 + 1, 16))
    assert field["u"].shape == (expected_windows, expected_windows)
    assert field["x"].shape == field["u"].shape
    # No motion between identical frames.
    assert np.abs(field["u"]).max() < 0.5
    assert np.abs(field["v"]).max() < 0.5


def test_piv_field_shape_mismatch_raises():
    a = np.zeros((100, 100))
    b = np.zeros((50, 50))
    try:
        compute_piv_field(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_piv_motion_signal_length_and_zero_for_static_video():
    frames = np.stack([_speckle_image()] * 5, axis=0)
    signal = compute_piv_motion_signal(frames, window_size=32, step=16)
    assert len(signal) == 4
    assert np.all(signal < 0.5)


def test_piv_motion_signal_detects_periodic_shift():
    base = _speckle_image()
    fps = 30.0
    n_frames = 90
    frames = np.zeros((n_frames, *base.shape), dtype=np.uint8)
    for i in range(n_frames):
        t = i / fps
        shift_amount = 2.0 * max(0.0, np.sin(2 * np.pi * 1.0 * t))
        shifted = ndi.shift(base, shift=(0, shift_amount), mode="reflect", order=3)
        frames[i] = np.clip(shifted, 0, 255).astype(np.uint8)

    signal = compute_piv_motion_signal(frames, window_size=32, step=16)
    assert len(signal) == n_frames - 1
    assert signal.max() > signal.mean() * 1.5  # visible peaks, not flat noise


def test_assess_texture_flags_flat_frame_as_low_texture():
    frame = np.full((160, 160), 100, dtype=np.uint8)
    result = assess_texture(frame, window_size=32, step=16)
    assert result["median_window_std"] == 0.0
    assert result["low_texture"] is True


def test_assess_texture_does_not_flag_speckled_frame():
    frame = _speckle_image(size=160)
    result = assess_texture(frame, window_size=32, step=16)
    assert result["median_window_std"] > 3.0
    assert result["low_texture"] is False


def test_compute_window_texture_mask_flags_flat_half_and_not_speckled_half():
    frame = _speckle_image(size=64)
    frame = frame.copy()
    frame[:, :32] = 100.0  # left half flat/untextured, right half speckle
    mask = compute_window_texture_mask(frame, window_size=32, step=32)
    assert mask.shape == (2, 2)
    assert not mask[:, 0].any()  # left column of windows: flat -> masked out
    assert mask[:, 1].all()  # right column: speckle -> kept


def test_compute_piv_motion_signal_masks_out_flat_region_noise():
    # A frame where only the left half has real periodic motion (speckle
    # shifting) and the right half is flat/static — without masking, the
    # flat half's near-zero-correlation-peak noise dilutes the true signal
    # from the moving half; with masking, only the moving half counts.
    base = _speckle_image(size=64)
    fps = 30.0
    n_frames = 60
    frames = np.zeros((n_frames, 64, 64), dtype=np.uint8)
    for i in range(n_frames):
        t = i / fps
        shift_amount = 3.0 * max(0.0, np.sin(2 * np.pi * 1.0 * t))
        shifted = ndi.shift(base, shift=(0, shift_amount), mode="reflect", order=3)
        combined = shifted.copy()
        combined[:, 32:] = 100.0  # right half stays flat/untextured throughout
        frames[i] = np.clip(combined, 0, 255).astype(np.uint8)

    mask = compute_window_texture_mask(frames[0], window_size=32, step=32)
    masked_signal = compute_piv_motion_signal(frames, window_size=32, step=32, texture_mask=mask)
    unmasked_signal = compute_piv_motion_signal(frames, window_size=32, step=32, texture_mask=None)

    # The masked signal (real motion only) should show a clearer peak-to-mean
    # ratio than the unmasked one, which is diluted/noised by the flat half.
    masked_ratio = masked_signal.max() / (masked_signal.mean() + 1e-9)
    unmasked_ratio = unmasked_signal.max() / (unmasked_signal.mean() + 1e-9)
    assert masked_ratio >= unmasked_ratio


def test_compute_piv_motion_signal_falls_back_to_unmasked_when_mask_is_all_false():
    base = _speckle_image(size=64)
    frames = np.stack([base] * 5, axis=0).astype(np.uint8)
    all_false_mask = np.zeros((2, 2), dtype=bool)
    signal_with_empty_mask = compute_piv_motion_signal(
        frames, window_size=32, step=32, texture_mask=all_false_mask
    )
    signal_without_mask = compute_piv_motion_signal(frames, window_size=32, step=32, texture_mask=None)
    assert np.allclose(signal_with_empty_mask, signal_without_mask)
