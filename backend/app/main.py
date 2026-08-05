from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.analysis import plotting
from app.analysis.beating import analyze_beating
from app.analysis.calcium import analyze_calcium
from app.analysis.morphology import analyze_morphology_2d, analyze_morphology_3d
from app.config import FRONTEND_DIR, MAX_FRAMES, MAX_UPLOAD_BYTES, RESULTS_DIR, UPLOADS_DIR
from app.utils.video_io import VideoLoadError, read_video_frames

app = FastAPI(title="Cardiomyocyte Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


@app.get("/api/health")
def health():
    return {"status": "ok"}


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds max upload size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            out.write(chunk)
    return dest


def _load_frames(path: Path, fps_override: float | None):
    try:
        frames, fps = read_video_frames(str(path), grayscale=True, max_frames=MAX_FRAMES)
    except VideoLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if fps_override:
        fps = fps_override
    return frames, fps


def _new_result_dir() -> tuple[str, Path]:
    result_id = uuid.uuid4().hex
    result_dir = RESULTS_DIR / result_id
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_id, result_dir


def _urls(result_id: str, result_dir: Path) -> dict:
    return {
        name: f"/results/{result_id}/{path.name}"
        for name, path in {
            "plot": result_dir / "plot.png",
            "csv": result_dir / "data.csv",
            "summary": result_dir / "summary.json",
        }.items()
        if path.exists()
    }


@app.post("/api/analyze/beating")
async def analyze_beating_endpoint(
    file: UploadFile = File(...),
    fps_override: float | None = Form(default=None),
    min_bpm_gap: float = Form(default=300.0),
    prominence_frac: float = Form(default=0.15),
    signal_mode: Literal["reference", "consecutive"] = Form(default="reference"),
    reference_index: int | None = Form(default=None),
):
    upload_path = _save_upload(file)
    try:
        frames, fps = _load_frames(upload_path, fps_override)
        result = analyze_beating(
            frames,
            fps,
            min_bpm_gap=min_bpm_gap,
            prominence_frac=prominence_frac,
            signal_mode=signal_mode,
            reference_index=reference_index,
        )
    finally:
        upload_path.unlink(missing_ok=True)

    result_id, result_dir = _new_result_dir()
    result.beats_df.to_csv(result_dir / "data.csv", index=False)
    (result_dir / "summary.json").write_text(json.dumps(result.summary, indent=2))
    plotting.plot_beating(result, str(result_dir / "plot.png"))

    return {"result_id": result_id, "summary": result.summary, "urls": _urls(result_id, result_dir)}


@app.post("/api/analyze/calcium")
async def analyze_calcium_endpoint(
    file: UploadFile = File(...),
    fps_override: float | None = Form(default=None),
    min_transients_per_min: float = Form(default=240.0),
    prominence_frac: float = Form(default=0.2),
):
    upload_path = _save_upload(file)
    try:
        frames, fps = _load_frames(upload_path, fps_override)
        result = analyze_calcium(
            frames,
            fps,
            min_transients_per_min=min_transients_per_min,
            prominence_frac=prominence_frac,
        )
    finally:
        upload_path.unlink(missing_ok=True)

    result_id, result_dir = _new_result_dir()
    result.transients_df.to_csv(result_dir / "data.csv", index=False)
    (result_dir / "summary.json").write_text(json.dumps(result.summary, indent=2))
    plotting.plot_calcium(result, str(result_dir / "plot.png"))

    return {"result_id": result_id, "summary": result.summary, "urls": _urls(result_id, result_dir)}


@app.post("/api/analyze/morphology")
async def analyze_morphology_endpoint(
    file: UploadFile = File(...),
    mode: Literal["2d", "3d"] = Form(default="2d"),
    min_object_size: int = Form(default=30),
):
    upload_path = _save_upload(file)
    try:
        frames, _fps = _load_frames(upload_path, None)
        if mode == "2d":
            result = analyze_morphology_2d(frames, min_object_px=min_object_size)
        else:
            result = analyze_morphology_3d(frames, min_object_voxels=min_object_size)
    finally:
        upload_path.unlink(missing_ok=True)

    result_id, result_dir = _new_result_dir()
    result.objects_df.to_csv(result_dir / "data.csv", index=False)
    (result_dir / "summary.json").write_text(json.dumps(result.summary, indent=2))
    plotting.plot_morphology(result, str(result_dir / "plot.png"))

    return {"result_id": result_id, "summary": result.summary, "urls": _urls(result_id, result_dir)}


# Mounted last so it never shadows the /api/* and /results/* routes above.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
