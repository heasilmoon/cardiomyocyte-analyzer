"""Make a small, shareable copy of a microscopy video.

Real recordings are often hundreds of MB. For sharing a representative
clip (e.g. to check what the field of view looks like, or to reproduce a
bug) a few seconds at reduced resolution is enough and comes out at a few
MB. Uses OpenCV only (already a dependency), no ffmpeg needed.

Usage (from backend/, with the venv active):
    python tools/shrink_video.py INPUT.mp4 OUTPUT.mp4 [--seconds 5] [--width 320]
        [--start 0] [--keep-every 1]

--seconds     length of the clip to keep (default 5 s)
--start       where in the video to start, in seconds (default 0)
--width       output width in pixels; height keeps the aspect ratio (default 320)
--keep-every  keep every N-th frame, e.g. 2 halves the frame rate (default 1)

Note: the output is re-encoded (MPEG-4), so use it for looking at the
footage and for rough checks — analyze the original for real results.
"""
from __future__ import annotations

import argparse
import os

import cv2


def shrink(
    src: str, dst: str, seconds: float = 5.0, width: int = 320, start: float = 0.0, keep_every: int = 1
) -> dict:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = width / in_w if in_w > width else 1.0
    out_w = int(round(in_w * scale))
    out_h = int(round(in_h * scale))
    out_fps = fps / keep_every

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
    max_frames = int(round(seconds * fps))
    read = written = 0
    while read < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if read % keep_every == 0:
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
        read += 1
    cap.release()
    writer.release()
    return {
        "input": f"{in_w}x{in_h} @ {fps:.1f} fps",
        "output": f"{out_w}x{out_h} @ {out_fps:.1f} fps, {written} frames ({written / out_fps:.1f} s)",
        "output_size_mb": os.path.getsize(dst) / (1024 * 1024),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--keep-every", type=int, default=1)
    args = p.parse_args()
    info = shrink(args.input, args.output, args.seconds, args.width, args.start, args.keep_every)
    for k, v in info.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
