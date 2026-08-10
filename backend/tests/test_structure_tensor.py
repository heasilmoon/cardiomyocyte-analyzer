import numpy as np

from app.analysis.structure_tensor import structure_tensor_alignment_2d, structure_tensor_alignment_3d


def _make_stripe_image(gradient_angle_deg, size=200, period=10.0):
    theta = np.radians(gradient_angle_deg)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    return (128 + 100 * np.sin(2 * np.pi * proj / period)).astype(np.uint8)


def test_2d_orientation_matches_known_stripe_angle():
    # Stripes are perpendicular to the intensity gradient, so a gradient
    # angle of 0 deg means fiber/stripe direction of -90 deg (axial wrap).
    for gradient_angle_deg, expected_fiber_deg in [(0, -90), (45, -45), (90, 0)]:
        img = _make_stripe_image(gradient_angle_deg)
        result = structure_tensor_alignment_2d(img, noise_sigma=1.0, integration_sigma=4.0)
        assert result["alignment_score"] > 0.95
        assert abs(result["mean_orientation_deg"] - expected_fiber_deg) < 2.0


def test_2d_alignment_low_for_random_noise():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (200, 200)).astype(np.uint8)
    result = structure_tensor_alignment_2d(noise, noise_sigma=1.0, integration_sigma=4.0)
    assert result["alignment_score"] < 0.3


def test_2d_orientation_and_coherence_maps_match_image_shape():
    img = _make_stripe_image(30)
    result = structure_tensor_alignment_2d(img)
    assert result["orientation_map"].shape == img.shape
    assert result["coherence_map"].shape == img.shape
    assert result["mean_coherence"] > 0.5


def test_3d_alignment_high_for_structured_volume():
    v = np.array([0.0, 1.0, 0.0])  # gradient purely along y
    size = 60
    zz, yy, xx = np.mgrid[0:20, 0:size, 0:size].astype(float)
    proj = zz * v[0] + yy * v[1] + xx * v[2]
    vol = (128 + 100 * np.sin(2 * np.pi * proj / 8.0)).astype(np.uint8)
    result = structure_tensor_alignment_3d(vol, noise_sigma=1.0, integration_sigma=3.0, stride=2)
    assert result["alignment_score_3d"] > 0.9
    # fiber direction must be perpendicular to the gradient direction v
    direction = np.array(result["mean_direction_zyx"])
    assert abs(np.dot(direction, v)) < 0.1


def test_3d_alignment_low_for_random_noise():
    rng = np.random.default_rng(1)
    noise_vol = rng.integers(0, 255, (20, 60, 60)).astype(np.uint8)
    result = structure_tensor_alignment_3d(noise_vol, noise_sigma=1.0, integration_sigma=3.0, stride=2)
    assert result["alignment_score_3d"] < 0.3


def test_3d_maps_are_downsampled_by_stride():
    vol = np.random.default_rng(2).integers(0, 255, (20, 40, 40)).astype(np.uint8)
    result = structure_tensor_alignment_3d(vol, stride=4)
    assert result["principal_direction_map"].shape == (5, 10, 10, 3)
    assert result["fa_map"].shape == (5, 10, 10)
