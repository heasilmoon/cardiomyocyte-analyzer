"""Video/frame-sequence loading shared by all analysis modules."""
from __future__ import annotations

import cv2
import numpy as np


class VideoLoadError(ValueError):
    pass


def read_video_frames(
    path: str,
    grayscale: bool = True,
    max_frames: int | None = None,
) -> tuple[np.ndarray, float]:
    """Read an mp4 (or any OpenCV-readable video) into a numpy array.

    Returns (frames, fps) where frames has shape (N, H, W) if grayscale
    else (N, H, W, 3). fps falls back to 30.0 when the container doesn't
    report a valid frame rate (common for re-encoded/screen-captured mp4s).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoLoadError(f"Could not open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:
        fps = 30.0

    frames = []
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()

    if not frames:
        raise VideoLoadError(f"No frames could be decoded from: {path}")

    return np.stack(frames, axis=0), float(fps)
