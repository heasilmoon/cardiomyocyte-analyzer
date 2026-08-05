"""Cell / structure morphology analysis.

2D mode: segments a single representative image (max-intensity projection
across the supplied frames, which collapses either a short static clip or
noise-y single shot into one clean image) and reports per-object shape
metrics via connected-component labeling.

3D mode: treats the frame sequence itself as a z-stack (frame index = z
slice) and performs the equivalent segmentation in 3D, reporting per-object
volume alongside the 2D shape metrics.

Pixel/voxel sizes are unknown from the video alone, so all size metrics are
reported in pixel / voxel units; the frontend can let a user supply a
calibration (um per pixel) to rescale downstream if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects


@dataclass
class MorphologyResult:
    mode: str
    n_objects: int
    objects_df: pd.DataFrame
    projection: np.ndarray
    label_image: np.ndarray
    summary: dict = field(default_factory=dict)


def _segment(image: np.ndarray, min_size: int) -> np.ndarray:
    threshold = threshold_otsu(image)
    binary = image > threshold
    binary = ndi.binary_fill_holes(binary)
    binary = remove_small_objects(binary, min_size=min_size)
    return label(binary)


def analyze_morphology_2d(frames: np.ndarray, min_object_px: int = 30) -> MorphologyResult:
    projection = frames.max(axis=0) if frames.ndim == 3 else frames
    label_image = _segment(projection, min_object_px)

    props = regionprops_table(
        label_image,
        intensity_image=projection,
        properties=(
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "equivalent_diameter_area",
            "mean_intensity",
            "centroid",
        ),
    )
    objects_df = pd.DataFrame(props)

    summary = {
        "n_objects": int(objects_df.shape[0]),
        "mean_area_px": float(objects_df["area"].mean()) if len(objects_df) else None,
        "median_area_px": float(objects_df["area"].median()) if len(objects_df) else None,
        "mean_eccentricity": float(objects_df["eccentricity"].mean()) if len(objects_df) else None,
        "total_covered_area_px": float(objects_df["area"].sum()) if len(objects_df) else 0.0,
        "image_area_px": int(projection.shape[0] * projection.shape[1]),
    }
    if summary["image_area_px"]:
        summary["coverage_fraction"] = summary["total_covered_area_px"] / summary["image_area_px"]

    return MorphologyResult(
        mode="2d",
        n_objects=int(objects_df.shape[0]),
        objects_df=objects_df,
        projection=projection,
        label_image=label_image,
        summary=summary,
    )


def analyze_morphology_3d(frames: np.ndarray, min_object_voxels: int = 100) -> MorphologyResult:
    """frames: (Z, H, W) grayscale stack, one z-slice per frame."""
    volume = frames
    threshold = threshold_otsu(volume)
    binary = volume > threshold
    binary = ndi.binary_fill_holes(binary)
    binary = remove_small_objects(binary, min_size=min_object_voxels)
    label_volume = label(binary)

    props = regionprops_table(
        label_volume,
        intensity_image=volume,
        properties=(
            "label",
            "area",  # voxel count in 3D
            "equivalent_diameter_area",
            "mean_intensity",
            "centroid",
            "bbox",
        ),
    )
    objects_df = pd.DataFrame(props)
    if "area" in objects_df:
        objects_df = objects_df.rename(columns={"area": "volume_voxels"})

    summary = {
        "n_objects": int(objects_df.shape[0]),
        "mean_volume_voxels": float(objects_df["volume_voxels"].mean()) if len(objects_df) else None,
        "median_volume_voxels": float(objects_df["volume_voxels"].median()) if len(objects_df) else None,
        "total_volume_voxels": float(objects_df["volume_voxels"].sum()) if len(objects_df) else 0.0,
        "stack_shape_zyx": list(volume.shape),
    }

    projection = volume.max(axis=0)
    label_projection = label_volume.max(axis=0)

    return MorphologyResult(
        mode="3d",
        n_objects=int(objects_df.shape[0]),
        objects_df=objects_df,
        projection=projection,
        label_image=label_projection,
        summary=summary,
    )
