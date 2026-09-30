import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"

for d in (STORAGE_DIR, UPLOADS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


# Safety limits so a huge upload can't exhaust memory when decoded frame-by-
# frame into a numpy array. All three can be changed with environment
# variables (e.g. in Render's Environment tab or before `uvicorn`):
#   MAX_UPLOAD_MB    per-file upload limit (default 1024 MB)
#   MAX_FRAMES       frames analyzed per video (default 3000 = 50 s @ 60 fps)
#   MAX_FRAME_SIDE   longer frame side, px; bigger videos are downscaled
#                    with area averaging before analysis (default 720).
#                    0 disables downscaling.
MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_MB", 1024) * 1024 * 1024
MAX_FRAMES = _env_int("MAX_FRAMES", 3000)
MAX_FRAME_SIDE = _env_int("MAX_FRAME_SIDE", 720)

# Local result files (plots, CSV, JSON) are pruned oldest-first once they
# exceed this total, so a small hosted disk (Render / Hugging Face Spaces)
# never fills up. Results mirrored to Supabase are unaffected. 0 disables.
RESULTS_MAX_BYTES = _env_int("RESULTS_MAX_MB", 500) * 1024 * 1024
# Refuse an upload (with a clear message) when the disk has less free space
# than the file plus this margin, instead of crashing mid-write.
MIN_FREE_DISK_BYTES = _env_int("MIN_FREE_DISK_MB", 200) * 1024 * 1024
