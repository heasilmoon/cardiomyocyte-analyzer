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
