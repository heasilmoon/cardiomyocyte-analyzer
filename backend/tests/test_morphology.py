import numpy as np

from app.analysis.morphology import analyze_morphology_2d, analyze_morphology_3d


def _make_blob_frames(n_frames=5, size=60):
    frames = np.zeros((n_frames, size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy, r in [(15, 15, 8), (45, 45, 10)]:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        frames[:, mask] = 200
    return frames


def _make_touching_blob_frames(n_frames=5, size=100):
    """Two overlapping circles, like real touching cells that a plain
    connected-component labeler would merge into a single object."""
    frames = np.zeros((n_frames, size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy, r in [(35, 50, 15), (60, 50, 15)]:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        frames[:, mask] = 200
    return frames


def _make_aligned_ellipse_frame(size=240, angle_deg=30.0, jitter_deg=0.0, seed=0):
    """A grid of small, widely-spaced ellipses all tilted the same way (or
    randomly if jitter_deg is large), for testing the alignment_score
    metric. Spacing is generous relative to ellipse size so none touch,
    isolating the alignment computation from the watershed-splitting logic
    tested elsewhere in this file."""
    import cv2

    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8)
    centers = [(x, y) for x in (60, 120, 180) for y in (60, 120, 180)]
    for cx, cy in centers:
        angle = angle_deg + rng.uniform(-jitter_deg, jitter_deg)
        cv2.ellipse(img, (cx, cy), (15, 5), angle, 0, 360, 200, -1)
    return img[None, :, :]  # (1, H, W) so it works as a single-frame stack


def test_analyze_morphology_2d_counts_objects():
    frames = _make_blob_frames()
    result = analyze_morphology_2d(frames, min_object_px=10)
    assert result.n_objects == 2
    assert set(result.objects_df.columns) >= {"area", "perimeter", "eccentricity"}


def test_analyze_morphology_3d_counts_objects():
    frames = _make_blob_frames(n_frames=8)
    result = analyze_morphology_3d(frames, min_object_voxels=50)
    assert result.n_objects == 2
    assert "volume_voxels" in result.objects_df.columns
    assert result.summary["stack_shape_zyx"] == [8, 60, 60]


def test_watershed_splits_touching_cells_2d():
    frames = _make_touching_blob_frames()
    merged = analyze_morphology_2d(frames, min_object_px=30, separate_touching=False)
    assert merged.n_objects == 1  # plain connected-component labeling merges them

    split = analyze_morphology_2d(
        frames, min_object_px=30, separate_touching=True, separation_min_distance=10
    )
    assert split.n_objects == 2  # watershed separates the two touching cells


def test_watershed_does_not_over_split_separate_flat_objects_3d():
    # Two non-touching, disc-like (thin in z) blobs should stay as 2 objects,
    # not get spuriously split by a too-small default seed spacing.
    frames = _make_blob_frames(n_frames=8)
    result = analyze_morphology_3d(frames, min_object_voxels=50, separate_touching=True)
    assert result.n_objects == 2


def test_alignment_score_high_for_uniformly_oriented_cells():
    frame = _make_aligned_ellipse_frame(angle_deg=30.0, jitter_deg=0.0)
    result = analyze_morphology_2d(frame, min_object_px=20)
    assert result.n_objects == 9
    assert result.summary["alignment_score"] is not None
    assert result.summary["alignment_score"] > 0.9


def test_alignment_score_low_for_randomly_oriented_cells():
    frame = _make_aligned_ellipse_frame(angle_deg=0.0, jitter_deg=90.0, seed=1)
    result = analyze_morphology_2d(frame, min_object_px=20)
    assert result.n_objects == 9
    assert result.summary["alignment_score"] is not None
    assert result.summary["alignment_score"] < 0.5
