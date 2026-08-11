import numpy as np
from scipy import ndimage as ndi

from app.analysis.piv import assess_texture, compute_piv_field, compute_piv_motion_signal


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
