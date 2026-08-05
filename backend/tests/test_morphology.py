import numpy as np

from app.analysis.morphology import analyze_morphology_2d, analyze_morphology_3d


def _make_blob_frames(n_frames=5, size=60):
    frames = np.zeros((n_frames, size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy, r in [(15, 15, 8), (45, 45, 10)]:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        frames[:, mask] = 200
    return frames


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
