import numpy as np

from app.main import _apply_roi


def test_apply_roi_returns_unmodified_frames_when_no_roi_given():
    frames = np.zeros((5, 20, 30), dtype=np.uint8)
    cropped, applied = _apply_roi(frames, None, None, None, None)
    assert cropped is frames
    assert applied is None


def test_apply_roi_crops_to_requested_region():
    frames = np.arange(5 * 20 * 30, dtype=np.uint8).reshape(5, 20, 30)
    cropped, applied = _apply_roi(frames, 5, 4, 10, 8)
    assert applied == {"x": 5, "y": 4, "w": 10, "h": 8}
    assert cropped.shape == (5, 8, 10)
    assert np.array_equal(cropped, frames[:, 4:12, 5:15])


def test_apply_roi_clamps_to_frame_bounds():
    frames = np.zeros((3, 20, 30), dtype=np.uint8)
    cropped, applied = _apply_roi(frames, 25, 15, 20, 20)
    assert applied == {"x": 25, "y": 15, "w": 5, "h": 5}
    assert cropped.shape == (3, 5, 5)


def test_apply_roi_clamps_negative_origin():
    # roi starting at -5 with width 10 only overlaps [0, 5) of the frame.
    frames = np.zeros((3, 20, 30), dtype=np.uint8)
    cropped, applied = _apply_roi(frames, -5, -5, 10, 10)
    assert applied == {"x": 0, "y": 0, "w": 5, "h": 5}
    assert cropped.shape == (3, 5, 5)


def test_apply_roi_scales_original_coordinates_to_downscaled_frames():
    # Original video 200x100, analyzed at half size (100x50): an ROI drawn on
    # the full-size preview at (40, 20) 80x40 must crop (20, 10) 40x20.
    frames = np.zeros((3, 50, 100), dtype=np.uint8)
    cropped, applied = _apply_roi(frames, 40, 20, 80, 40, scale=0.5)
    assert cropped.shape == (3, 20, 40)
    assert (applied["x"], applied["y"], applied["w"], applied["h"]) == (40, 20, 80, 40)
    assert applied["analysis_px"] == {"x": 20, "y": 10, "w": 40, "h": 20}


def test_prune_results_removes_oldest_until_under_cap(tmp_path):
    import os
    import time

    from app.main import _prune_results

    for i in range(4):
        d = tmp_path / f"r{i}"
        d.mkdir()
        (d / "plot.png").write_bytes(b"x" * 1000)
        t = time.time() - (10 - i) * 100  # r0 oldest
        os.utime(d, (t, t))
    removed = _prune_results(max_bytes=2500, results_dir=tmp_path)
    assert removed == 2
    assert not (tmp_path / "r0").exists() and not (tmp_path / "r1").exists()
    assert (tmp_path / "r2").exists() and (tmp_path / "r3").exists()
    assert _prune_results(max_bytes=0, results_dir=tmp_path) == 0  # disabled
