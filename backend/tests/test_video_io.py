import cv2
import numpy as np

from app.utils.video_io import read_video_frames, read_video_frames_scaled


def _write_video(path, n_frames=12, w=640, h=360, fps=30.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_read_video_frames_scaled_downscales_long_side(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, w=640, h=360)
    frames, fps, scale = read_video_frames_scaled(str(p), max_side=320)
    assert frames.shape[1:] == (180, 320)
    assert abs(scale - 0.5) < 1e-9
    assert fps == 30.0

    full, _, scale_full = read_video_frames_scaled(str(p), max_side=None)
    assert full.shape[1:] == (360, 640) and scale_full == 1.0
    # Videos already within the limit are untouched.
    same, _, s2 = read_video_frames_scaled(str(p), max_side=1000)
    assert same.shape == full.shape and s2 == 1.0


def test_read_video_frames_keeps_old_signature(tmp_path):
    p = tmp_path / "v.mp4"
    _write_video(p, n_frames=5)
    frames, fps = read_video_frames(str(p), max_frames=3)
    assert frames.shape[0] == 3
