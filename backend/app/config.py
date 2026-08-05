from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"

for d in (STORAGE_DIR, UPLOADS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Safety limits so a huge upload can't exhaust container memory when
# decoded frame-by-frame into a numpy array.
MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # 300 MB
MAX_FRAMES = 3000
