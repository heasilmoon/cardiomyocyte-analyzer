"""Generate small synthetic mp4s for smoke-testing the analysis pipelines.

Not a real cardiomyocyte recording — just a pulsing blob (for beating /
morphology) and a flickering-intensity blob (for calcium) so the endpoints
and signal-processing code can be exercised end to end without real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def make_beating_video(path: str, fps: int = 30, duration_s: int = 6, bpm: int = 60):
    w, h = 160, 160
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_frames = fps * duration_s
    beat_hz = bpm / 60.0
    for i in range(n_frames):
        t = i / fps
        radius = 30 + 8 * max(0.0, np.sin(2 * np.pi * beat_hz * t)) ** 3
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(frame, (w // 2, h // 2), int(radius), (200, 200, 200), -1)
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def make_calcium_video(path: str, fps: int = 30, duration_s: int = 6, hz: float = 1.0):
    w, h = 160, 160
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_frames = fps * duration_s
    for i in range(n_frames):
        t = i / fps
        phase = (t * hz) % 1.0
        intensity = 60 + 150 * np.exp(-phase * 8) * (1 if phase < 0.15 else np.exp(-(phase - 0.15) * 6))
        intensity = 60 + 150 * max(0.0, np.exp(-((phase) * 10)) if phase < 0.5 else 0.0)
        val = int(np.clip(60 + 150 * np.exp(-((phase * 6) ** 2)), 0, 255))
        frame = np.full((h, w, 3), val, dtype=np.uint8)
        noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def make_morphology_video(path: str, fps: int = 5, n_frames: int = 8, n_cells: int = 12):
    w, h = 200, 200
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    centers = rng.integers(20, w - 20, size=(n_cells, 2))
    radii = rng.integers(6, 14, size=n_cells)
    for _ in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for (cx, cy), r in zip(centers, radii):
            cv2.circle(frame, (int(cx), int(cy)), int(r), (180, 180, 180), -1)
        noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    make_beating_video(str(out_dir / "beating.mp4"))
    make_calcium_video(str(out_dir / "calcium.mp4"))
    make_morphology_video(str(out_dir / "morphology.mp4"))
    print("wrote synthetic videos to", out_dir)
