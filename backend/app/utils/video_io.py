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


def extract_first_frame_png(path: str) -> bytes:
    """Decode just the first frame of a video and PNG-encode it.

    Used for the frontend's ROI-selection preview. Browsers can't reliably
    decode every codec OpenCV can — e.g. this project's own synthetic test
    fixtures are MPEG-4 Part 2 ("mp4v" fourcc), which Chromium's <video>
    element silently fails to decode — so the preview is generated through
    the same OpenCV decode path the actual analysis uses, rather than
    relying on native browser video decoding.
    """
    frames, _ = read_video_frames(path, grayscale=False, max_frames=1)
    frame_bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise VideoLoadError("Could not encode preview frame")
    return buf.tobytes()
