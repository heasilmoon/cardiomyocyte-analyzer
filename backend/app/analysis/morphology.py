"""Cell / structure morphology analysis.

2D mode: segments a single representative image (max-intensity projection
across the supplied frames, which collapses either a short static clip or
noise-y single shot into one clean image) and reports per-object shape
metrics, including a population alignment score.

3D mode: treats the frame sequence itself as a z-stack (frame index = z
slice) and performs the equivalent segmentation in 3D, reporting per-object
volume and 3D alignment alongside the 2D shape metrics.

Touching objects are split via distance-transform watershed (works
identically in 2D and 3D since scipy/skimage's distance transform, peak
finding, and watershed are all dimension-agnostic) rather than left merged
under a single connected-component label.

Pixel/voxel sizes are unknown from the video alone, so all size metrics are
reported in pixel / voxel units; the frontend can let a user supply a
calibration (um per pixel) to rescale downstream if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


@dataclass
class MorphologyResult:
    mode: str
    n_objects: int
    objects_df: pd.DataFrame
    projection: np.ndarray
    label_image: np.ndarray
    summary: dict = field(default_factory=dict)


def _segment(
    image: np.ndarray,
    min_size: int,
    separate_touching: bool = True,
    min_distance: int = 10,
) -> np.ndarray:
    """Threshold + label, optionally splitting touching objects via watershed.

    Works for both 2D images and 3D volumes: scipy's distance transform and
    skimage's peak_local_max / watershed are all dimension-agnostic, so the
    same code path segments a z-stack exactly like a flat image.

    min_distance is the minimum expected spacing between two distinct
    objects' centers, in pixels/voxels — NOT derived from min_size (an
    object big enough to keep can still be much smaller than the typical
    object, e.g. a partially-imaged cell at the frame edge), so it needs
    its own sensible default. Too small a value causes spurious splits:
    a single blob's distance-transform interior is rarely a single sharp
    point, especially for the flat/disc-like objects typical of a
    video-frames-as-z-slices "3D" stack, and a couple of near-tied local
    maxima a few pixels apart in that plateau will otherwise be treated as
    two separate objects. Tune this to roughly the smallest expected
    center-to-center spacing between real, distinct objects.
    """
    threshold = threshold_otsu(image)
    binary = image > threshold
    binary = ndi.binary_fill_holes(binary)
    binary = remove_small_objects(binary, min_size=min_size)

    if not binary.any():
        return label(binary)

    if not separate_touching:
        return label(binary)

    distance = ndi.distance_transform_edt(binary)
    # Smooth only for peak-finding (not for the watershed elevation map
    # below) to further reduce spurious multi-peak plateaus.
    smoothed_distance = ndi.gaussian_filter(distance, sigma=1.0)
    coords = peak_local_max(smoothed_distance, min_distance=min_distance, labels=binary, exclude_border=False)
    if len(coords) == 0:
        return label(binary)

    markers = np.zeros(distance.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    markers = ndi.grey_dilation(markers, size=(3,) * image.ndim)
    markers[~binary] = 0
    return watershed(-distance, markers, mask=binary)


def _circular_alignment_2d(orientations_rad: np.ndarray) -> tuple[float | None, float | None]:
    """Population alignment score for axial (mod-pi) 2D orientation angles.

    Doubling the angle maps the axial [-pi/2, pi/2] range onto a full circle
    so the usual circular mean / resultant length works (a cell pointing at
    +85 deg and one at -85 deg are nearly parallel, not opposite). Returns
    (alignment_score in [0,1], mean_orientation_deg); 1 = perfectly aligned,
    0 = uniformly random orientation.
    """
    if len(orientations_rad) < 2:
        return None, None
    resultant = np.mean(np.exp(2j * orientations_rad))
    alignment_score = float(np.abs(resultant))
    mean_orientation_deg = float(np.degrees(0.5 * np.angle(resultant)))
    return alignment_score, mean_orientation_deg


def _principal_axis_3d(inertia_tensor: np.ndarray) -> np.ndarray:
    """Unit vector along an object's long axis from its inertia tensor.

    Moment of inertia about an axis is smallest when mass (voxels) sits
    close to that axis, so for an elongated object the eigenvector of the
    *smallest* eigenvalue points along the long axis.
    """
    eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
    return eigvecs[:, 0]


def _nematic_alignment_3d(directions: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    """Population alignment score for 3D axes (headless unit vectors).

    Standard nematic order-parameter construction (liquid-crystal / fiber
    alignment literature): Q = mean(3 n(x)n - I) / 2 is invariant to n -> -n,
    so it handles the sign ambiguity of a "long axis" correctly. Its largest
    eigenvalue S in [0,1] is the alignment score (0 = isotropic/random,
    1 = perfectly aligned) and the matching eigenvector is the mean
    alignment direction.
    """
    if len(directions) < 2:
        return None, None
    q_tensors = [3.0 * np.outer(n, n) - np.eye(3) for n in directions]
    q_mean = np.mean(q_tensors, axis=0) / 2.0
    eigvals, eigvecs = np.linalg.eigh(q_mean)
    order = np.argsort(eigvals)[::-1]
    s = float(max(eigvals[order[0]], 0.0))
    director = eigvecs[:, order[0]]
    return s, director


def analyze_morphology_2d(
    frames: np.ndarray,
    min_object_px: int = 30,
    separate_touching: bool = True,
    separation_min_distance: int = 10,
) -> MorphologyResult:
    projection = frames.max(axis=0) if frames.ndim == 3 else frames
    label_image = _segment(
        projection, min_object_px, separate_touching=separate_touching, min_distance=separation_min_distance
    )

    props = regionprops_table(
        label_image,
        intensity_image=projection,
        properties=(
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "orientation",
            "major_axis_length",
            "minor_axis_length",
            "equivalent_diameter_area",
            "mean_intensity",
            "centroid",
        ),
    )
    objects_df = pd.DataFrame(props)
    if "orientation" in objects_df:
        objects_df["orientation_deg"] = np.degrees(objects_df["orientation"])

    alignment_score, mean_orientation_deg = (
        _circular_alignment_2d(objects_df["orientation"].to_numpy()) if len(objects_df) else (None, None)
    )

    summary = {
        "n_objects": int(objects_df.shape[0]),
        "mean_area_px": float(objects_df["area"].mean()) if len(objects_df) else None,
        "median_area_px": float(objects_df["area"].median()) if len(objects_df) else None,
        "mean_eccentricity": float(objects_df["eccentricity"].mean()) if len(objects_df) else None,
        "total_covered_area_px": float(objects_df["area"].sum()) if len(objects_df) else 0.0,
        "image_area_px": int(projection.shape[0] * projection.shape[1]),
        "alignment_score": alignment_score,
        "mean_orientation_deg": mean_orientation_deg,
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


def analyze_morphology_3d(
    frames: np.ndarray,
    min_object_voxels: int = 100,
    separate_touching: bool = True,
    separation_min_distance: int = 10,
) -> MorphologyResult:
    """frames: (Z, H, W) grayscale stack, one z-slice per frame."""
    volume = frames
    label_volume = _segment(
        volume, min_object_voxels, separate_touching=separate_touching, min_distance=separation_min_distance
    )

    regions = regionprops(label_volume, intensity_image=volume)
    rows = []
    directions = []
    for region in regions:
        axis = _principal_axis_3d(region.inertia_tensor)
        directions.append(axis)
        rows.append(
            {
                "label": region.label,
                "volume_voxels": region.area,
                "equivalent_diameter_area": region.equivalent_diameter_area,
                "mean_intensity": region.mean_intensity,
                "centroid-0": region.centroid[0],
                "centroid-1": region.centroid[1],
                "centroid-2": region.centroid[2],
                "axis_z": axis[0],
                "axis_y": axis[1],
                "axis_x": axis[2],
            }
        )
    objects_df = pd.DataFrame(rows)

    alignment_score, director = _nematic_alignment_3d(np.array(directions)) if directions else (None, None)

    summary = {
        "n_objects": int(objects_df.shape[0]),
        "mean_volume_voxels": float(objects_df["volume_voxels"].mean()) if len(objects_df) else None,
        "median_volume_voxels": float(objects_df["volume_voxels"].median()) if len(objects_df) else None,
        "total_volume_voxels": float(objects_df["volume_voxels"].sum()) if len(objects_df) else 0.0,
        "stack_shape_zyx": list(volume.shape),
        "alignment_score_3d": alignment_score,
        "mean_direction_zyx": [float(v) for v in director] if director is not None else None,
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
