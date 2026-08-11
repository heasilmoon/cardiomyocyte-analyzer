from __future__ import annotations

import io
import json
import re
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pandas as pd

from app.analysis import plotting
from app.analysis.beating import analyze_beating
from app.analysis.calcium import analyze_calcium
from app.analysis.colocalization import analyze_colocalization
from app.analysis.group_stats import compare_groups
from app.analysis.morphology import analyze_morphology_2d, analyze_morphology_3d
from app.analysis.validation_stats import compute_agreement
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
    min_bpm_gap: float | None = Form(default=None),
    prominence_frac: float = Form(default=0.15),
    signal_mode: Literal["reference", "consecutive", "piv"] = Form(default="reference"),
    reference_index: int | None = Form(default=None),
    piv_window_size: int = Form(default=32),
    piv_step: int | None = Form(default=None),
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
            piv_window_size=piv_window_size,
            piv_step=piv_step,
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
    separate_touching: bool = Form(default=True),
    separation_min_distance: int = Form(default=10),
    compute_texture_alignment: bool = Form(default=False),
):
    upload_path = _save_upload(file)
    try:
        frames, _fps = _load_frames(upload_path, None)
        if mode == "2d":
            result = analyze_morphology_2d(
                frames,
                min_object_px=min_object_size,
                separate_touching=separate_touching,
                separation_min_distance=separation_min_distance,
                compute_texture_alignment=compute_texture_alignment,
            )
        else:
            result = analyze_morphology_3d(
                frames,
                min_object_voxels=min_object_size,
                separate_touching=separate_touching,
                separation_min_distance=separation_min_distance,
                compute_texture_alignment=compute_texture_alignment,
            )
    finally:
        upload_path.unlink(missing_ok=True)

    result_id, result_dir = _new_result_dir()
    result.objects_df.to_csv(result_dir / "data.csv", index=False)
    (result_dir / "summary.json").write_text(json.dumps(result.summary, indent=2))
    plotting.plot_morphology(result, str(result_dir / "plot.png"))

    return {"result_id": result_id, "summary": result.summary, "urls": _urls(result_id, result_dir)}


def _analyze_one(analysis_type: str, morphology_mode: str, frames, fps) -> dict:
    if analysis_type == "beating":
        return analyze_beating(frames, fps).summary
    if analysis_type == "calcium":
        return analyze_calcium(frames, fps).summary
    if morphology_mode == "2d":
        return analyze_morphology_2d(frames).summary
    return analyze_morphology_3d(frames).summary


async def _summarize_group(files: list[UploadFile], analysis_type: str, morphology_mode: str) -> list[dict]:
    summaries = []
    for f in files:
        path = _save_upload(f)
        try:
            frames, fps = _load_frames(path, None)
            summary = _analyze_one(analysis_type, morphology_mode, frames, fps)
            summaries.append({"filename": f.filename, **summary})
        finally:
            path.unlink(missing_ok=True)
    return summaries


@app.post("/api/analyze/batch")
async def analyze_batch_endpoint(
    analysis_type: Literal["beating", "calcium", "morphology"] = Form(...),
    morphology_mode: Literal["2d", "3d"] = Form(default="2d"),
    files: list[UploadFile] = File(...),
):
    """Run one analysis over many videos and return a single combined CSV.

    Meant for quickly generating this tool's values across an entire real
    dataset (e.g. every recording in a validation study), one row per video.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one video file is required")

    summaries = await _summarize_group(files, analysis_type, morphology_mode)

    result_id, result_dir = _new_result_dir()
    pd.DataFrame(summaries).to_csv(result_dir / "data.csv", index=False)
    (result_dir / "summary.json").write_text(json.dumps(summaries, indent=2))

    return {"result_id": result_id, "n_videos": len(summaries), "summaries": summaries, "urls": _urls(result_id, result_dir)}


def _parse_batch_labels(text: str | None, expected_count: int, field_name: str) -> list[str] | None:
    """Parse an optional newline/comma-separated batch (cluster) label list.

    One label per uploaded file, same order as the uploads. Used to fit the
    cluster-aware linear mixed-effects model in compare_groups instead of
    treating every video as an independent replicate — see group_stats.py.
    Returns None (no clustering) when the field is left blank.
    """
    if text is None or not text.strip():
        return None
    labels = [part.strip() for part in re.split(r"[\n,]+", text) if part.strip()]
    if len(labels) != expected_count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name}: expected {expected_count} batch/sample labels (one per uploaded file, "
                f"same order), got {len(labels)}"
            ),
        )
    return labels


@app.post("/api/analyze/compare")
async def analyze_compare_endpoint(
    analysis_type: Literal["beating", "calcium", "morphology"] = Form(...),
    morphology_mode: Literal["2d", "3d"] = Form(default="2d"),
    group_a_label: str = Form(default="Group A"),
    group_b_label: str = Form(default="Group B"),
    group_a_files: list[UploadFile] = File(...),
    group_b_files: list[UploadFile] = File(...),
    group_a_batches: str | None = Form(default=None),
    group_b_batches: str | None = Form(default=None),
):
    if not group_a_files or not group_b_files:
        raise HTTPException(status_code=400, detail="Both groups need at least one video file")

    clusters_a = _parse_batch_labels(group_a_batches, len(group_a_files), "group_a_batches")
    clusters_b = _parse_batch_labels(group_b_batches, len(group_b_files), "group_b_batches")

    summaries_a = await _summarize_group(group_a_files, analysis_type, morphology_mode)
    summaries_b = await _summarize_group(group_b_files, analysis_type, morphology_mode)
    comparison = compare_groups(
        summaries_a, summaries_b, group_a_label, group_b_label, clusters_a=clusters_a, clusters_b=clusters_b
    )

    result_id, result_dir = _new_result_dir()
    for s in summaries_a:
        s["group"] = group_a_label
    for s in summaries_b:
        s["group"] = group_b_label
    combined_df = pd.DataFrame(summaries_a + summaries_b)
    combined_df.to_csv(result_dir / "data.csv", index=False)
    (result_dir / "summary.json").write_text(json.dumps(comparison, indent=2))
    plotting.plot_group_comparison(comparison, str(result_dir / "plot.png"))

    return {"result_id": result_id, "comparison": comparison, "urls": _urls(result_id, result_dir)}


@app.post("/api/validate/agreement")
async def validate_agreement_endpoint(
    file: UploadFile = File(...),
    column_a: str = Form(...),
    column_b: str = Form(...),
    label_a: str = Form(default="This tool"),
    label_b: str = Form(default="Reference method"),
):
    """Method-agreement analysis from a CSV of paired (this-tool, reference)
    values — e.g. one row per video, one column from this tool's batch
    output and one column of matching Fiji/MUSCLEMOTION values."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    for col in (column_a, column_b):
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found in CSV. Available columns: {list(df.columns)}",
            )

    paired = df[[column_a, column_b]].apply(pd.to_numeric, errors="coerce").dropna()
    try:
        agreement = compute_agreement(paired[column_a].tolist(), paired[column_b].tolist())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result_id, result_dir = _new_result_dir()
    pd.DataFrame(
        {
            column_a: agreement["values_a"],
            column_b: agreement["values_b"],
            "diff": agreement["diffs"],
            "mean": agreement["means"],
        }
    ).to_csv(result_dir / "data.csv", index=False)
    stats_only = {k: v for k, v in agreement.items() if k not in ("values_a", "values_b", "diffs", "means")}
    (result_dir / "summary.json").write_text(json.dumps(stats_only, indent=2))
    plotting.plot_agreement(agreement, label_a, label_b, str(result_dir / "plot.png"))

    return {"result_id": result_id, "stats": stats_only, "urls": _urls(result_id, result_dir)}


def _load_projection(file: UploadFile):
    path = _save_upload(file)
    try:
        frames, _fps = _load_frames(path, None)
        return frames.max(axis=0) if frames.ndim == 3 else frames
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/analyze/colocalization")
async def analyze_colocalization_endpoint(
    channel_a_file: UploadFile = File(...),
    channel_b_file: UploadFile = File(...),
    label_a: str = Form(default="Channel A"),
    label_b: str = Form(default="Channel B"),
):
    """Two-channel colocalization from a pair of image/video uploads (each
    reduced to a max-intensity projection, same as morphology's 2D mode)."""
    projection_a = _load_projection(channel_a_file)
    projection_b = _load_projection(channel_b_file)
    try:
        coloc_stats = analyze_colocalization(projection_a, projection_b)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result_id, result_dir = _new_result_dir()
    (result_dir / "summary.json").write_text(json.dumps(coloc_stats, indent=2))
    plotting.plot_colocalization(
        projection_a, projection_b, coloc_stats, label_a, label_b, str(result_dir / "plot.png")
    )

    return {"result_id": result_id, "stats": coloc_stats, "urls": _urls(result_id, result_dir)}


# Mounted last so it never shadows the /api/* and /results/* routes above.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
