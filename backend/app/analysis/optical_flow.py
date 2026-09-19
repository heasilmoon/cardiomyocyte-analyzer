"""Dense optical-flow contractility analysis (ContractionWave-style).

Re-implements the measurement idea of CONTRACTIONWAVE (Scalzo et al., 2021,
Cell Reports Methods, doi:10.1016/j.crmeth.2021.100044): the Farneback dense
optical flow between consecutive frames gives a displacement vector for
every pixel; the mean vector magnitude per frame pair, multiplied by the
frame rate and the pixel size, is the "average speed" of the tissue (µm/s).
That speed-versus-time curve shows two waves per beat — the contraction
stroke and the relaxation stroke — and ContractionWave's contractility
parameters are read off those two waves:

  MCS / MRS   maximum contraction / relaxation speed (wave peaks)
  CTP         contraction time-to-peak (wave start -> MCS)
  CTPMS       contraction time from peak to minimum speed (MCS -> dip)
  CT          contraction time (wave start -> dip) = CTP + CTPMS
  RTP         relaxation time-to-peak (dip -> MRS)
  RTPB        relaxation time from peak to baseline (MRS -> wave end)
  RT          relaxation time (dip -> wave end) = RTP + RTPB
  CRT         contraction-relaxation time (wave start -> wave end) = CT + RT
  TBC-RMS     time between contraction and relaxation maximum speed
  CRA / SA    contraction-relaxation area / shortening area: integral of the
              speed curve over the whole wave / over the contraction phase
              (a path length, µm or px)

Only the *method* is reproduced here, from the paper and from reading how
the published tool computes its numbers; no ContractionWave code is used
(it is a GPL-2.0 Tkinter desktop application, this project is MIT).
Default Farneback parameters match ContractionWave's defaults.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from app.analysis.signal_common import detect_peaks

_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

# ContractionWave defaults for cv2.calcOpticalFlowFarneback.
FARNEBACK_DEFAULTS: dict = {
    "pyr_scale": 0.5,
    "levels": 1,
    "winsize": 15,
    "iterations": 1,
    "poly_n": 7,
    "poly_sigma": 1.5,
}


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    f = frame.astype(np.float32)
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.uint8)
    return ((f - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def compute_flow(frame_a: np.ndarray, frame_b: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Dense Farneback flow from frame_a to frame_b, shape (H, W, 2) in px."""
    p = {**FARNEBACK_DEFAULTS, **(params or {})}
    return cv2.calcOpticalFlowFarneback(
        _to_uint8(frame_a),
        _to_uint8(frame_b),
        None,
        float(p["pyr_scale"]),
        int(p["levels"]),
        int(p["winsize"]),
        int(p["iterations"]),
        int(p["poly_n"]),
        float(p["poly_sigma"]),
        0,
    )


def compute_optical_flow_speed_signal(
    frames: np.ndarray,
    fps: float,
    um_per_px: float | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Mean optical-flow speed per frame pair, length N-1.

    Units are µm/s when um_per_px is given (ContractionWave's "Average
    Speed"), otherwise px/s. Pixels are averaged over the whole field like
    ContractionWave does by default (no magnitude segmentation), so the ROI
    tool is the way to restrict the measurement to the cell/tissue.
    """
    if frames.ndim != 3:
        raise ValueError("Expected grayscale frames with shape (N, H, W)")
    n = frames.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 frames for optical flow")
    scale = float(fps) * (float(um_per_px) if um_per_px else 1.0)
    speeds = np.empty(n - 1, dtype=np.float64)
    prev = _to_uint8(frames[0])
    for i in range(n - 1):
        cur = _to_uint8(frames[i + 1])
        flow = compute_flow(prev, cur, params)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        speeds[i] = float(mag.mean()) * scale
        prev = cur
    return speeds


def flow_to_vector_grid(flow: np.ndarray, step: int = 8, scale: float = 1.0) -> dict:
    """Subsample a dense flow field to a coarse grid for quiver plotting,
    in the same {x, y, u, v} layout as piv.compute_piv_field."""
    h, w = flow.shape[:2]
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    x_grid, y_grid = np.meshgrid(xs, ys)
    u = flow[ys][:, xs, 0] * scale
    v = flow[ys][:, xs, 1] * scale
    return {"x": x_grid, "y": y_grid, "u": u, "v": v}


def _pair_strokes(
    speed: np.ndarray,
    anchors: np.ndarray,
    strokes: np.ndarray,
    baseline: float,
    period_frames: float,
    min_secondary_frac: float = 0.2,
    max_pair_gap_frac: float = 0.6,
) -> list[tuple[int, int | None]]:
    """Pair each beat anchor (the tallest stroke of a beat) with the other
    stroke of the same beat, giving (contraction_idx, relaxation_idx|None).

    The partner must be a stroke peak inside this beat's window (halfway to
    the neighbouring anchors), within max_pair_gap_frac of a period of the
    anchor, and at least min_secondary_frac of the anchor's height above
    baseline — the last rule keeps noise bumps in the diastolic interval
    from being mistaken for a relaxation wave. Whichever of the pair comes
    first is the contraction stroke (ContractionWave's convention).
    """
    n = len(speed)
    pairs: list[tuple[int, int | None]] = []
    for i, a in enumerate(anchors):
        lo = (anchors[i - 1] + a) // 2 if i > 0 else max(0, int(a - 0.75 * period_frames))
        hi = (a + anchors[i + 1]) // 2 if i + 1 < len(anchors) else min(n - 1, int(a + 0.75 * period_frames))
        anchor_height = speed[a] - baseline
        best = None
        for s in strokes:
            if s == a or s < lo or s > hi or abs(int(s) - int(a)) > max_pair_gap_frac * period_frames:
                continue
            if speed[s] - baseline < min_secondary_frac * anchor_height:
                continue
            if best is None or speed[s] > speed[best]:
                best = int(s)
        if best is None:
            pairs.append((int(a), None))
        elif best < a:
            pairs.append((best, int(a)))
        else:
            pairs.append((int(a), best))
    return pairs


def _walk_while_above(signal: np.ndarray, start: int, threshold: float, direction: int, bound: int) -> int:
    i = start
    while 0 <= i + direction < len(signal) and (i + direction) * direction <= bound * direction:
        if signal[i + direction] <= threshold:
            break
        i += direction
    return i


def analyze_speed_waves(
    time_s: np.ndarray,
    speed: np.ndarray,
    fps: float,
    period_s: float,
    prominence_frac: float = 0.15,
    threshold_frac: float = 0.10,
) -> dict:
    """ContractionWave-style wave analysis of a (smoothed) speed signal.

    Returns a dict with per-beat rows (cycles) and the index arrays used
    for plotting: contraction peaks, relaxation peaks, wave starts, ends.
    The wave baseline is the 10th percentile of the signal; a wave begins
    and ends where the speed crosses baseline + threshold_frac * (range),
    which mirrors ContractionWave's automatic-baseline detection and the
    10 %-above-baseline convention used for contraction duration by
    MUSCLEMOTION.
    """
    n = len(speed)
    period_frames = max(period_s * fps, 2.0)
    # Beat anchors: one peak per beat (the tallest stroke), using the same
    # period-derived spacing rule as the other signal modes so beat counts
    # stay comparable across modes.
    anchor_gap_bpm = float(np.clip(100.0 / period_s, 30.0, 400.0))
    anchors = np.asarray(
        detect_peaks(speed, fps, min_bpm_gap=anchor_gap_bpm, prominence_frac=prominence_frac), dtype=int
    )
    # Stroke peaks: finer spacing so both strokes of a beat can be found
    # (they are usually >= ~15 % of a period apart).
    stroke_gap_bpm = 60.0 * fps / max(0.15 * period_frames, 1.0)
    strokes = np.asarray(
        detect_peaks(speed, fps, min_bpm_gap=stroke_gap_bpm, prominence_frac=prominence_frac), dtype=int
    )

    lo = float(np.percentile(speed, 10))
    hi = float(np.percentile(speed, 99))
    threshold = lo + threshold_frac * max(hi - lo, 1e-12)

    empty = {
        "cycles_df": pd.DataFrame(),
        "contraction_idx": np.array([], dtype=int),
        "relaxation_idx": np.array([], dtype=int),
        "start_idx": np.array([], dtype=int),
        "end_idx": np.array([], dtype=int),
        "baseline_speed": lo,
        "wave_threshold": threshold,
    }
    if len(anchors) == 0:
        return empty

    pairs = _pair_strokes(speed, anchors, strokes, lo, period_frames)

    rows = []
    c_idx, r_idx, s_idx, e_idx = [], [], [], []
    dt = 1.0 / fps
    for k, (c, r) in enumerate(pairs):
        prev_last = pairs[k - 1][1] if (k > 0 and pairs[k - 1][1] is not None) else (pairs[k - 1][0] if k > 0 else None)
        left_bound = prev_last + 1 if prev_last is not None else 0
        right_bound = pairs[k + 1][0] - 1 if k + 1 < len(pairs) else n - 1
        start = _walk_while_above(speed, c, threshold, -1, left_bound)
        last_peak = r if r is not None else c
        end = _walk_while_above(speed, last_peak, threshold, +1, right_bound)
        complete = r is not None and end < right_bound
        if r is not None:
            dip = c + int(np.argmin(speed[c : r + 1]))
        else:
            dip = None

        def _area(a: int, b: int) -> float | None:
            if b <= a:
                return None
            return float(_trapezoid(np.clip(speed[a : b + 1] - lo, 0, None), dx=dt))

        mcs = float(speed[c] - lo)
        mrs = float(speed[r] - lo) if r is not None else None
        ctp = float(time_s[c] - time_s[start])
        ctpms = float(time_s[dip] - time_s[c]) if dip is not None else None
        ct = float(time_s[dip] - time_s[start]) if dip is not None else None
        rtp = float(time_s[r] - time_s[dip]) if r is not None else None
        rtpb = float(time_s[end] - time_s[r]) if (r is not None and end > r) else None
        rt = float(time_s[end] - time_s[dip]) if (dip is not None and end > dip) else None
        crt = float(time_s[end] - time_s[start]) if end > start else None
        rows.append(
            {
                "beat_index": k,
                "wave_start_time_s": float(time_s[start]),
                "contraction_peak_time_s": float(time_s[c]),
                "min_speed_time_s": float(time_s[dip]) if dip is not None else None,
                "relaxation_peak_time_s": float(time_s[r]) if r is not None else None,
                "wave_end_time_s": float(time_s[end]),
                "complete_wave": bool(complete),
                "max_contraction_speed": mcs,
                "max_relaxation_speed": mrs,
                "mcs_mrs_difference": (mcs - mrs) if mrs is not None else None,
                "contraction_time_to_peak_s": ctp,
                "contraction_peak_to_min_speed_s": ctpms,
                "contraction_time_s": ct,
                "relaxation_time_to_peak_s": rtp,
                "relaxation_peak_to_baseline_s": rtpb,
                "relaxation_time_s": rt,
                "contraction_relaxation_time_s": crt,
                "time_between_max_speeds_s": float(time_s[r] - time_s[c]) if r is not None else None,
                "contraction_relaxation_area": _area(start, end),
                "shortening_area": _area(start, dip) if dip is not None else None,
            }
        )
        c_idx.append(c)
        s_idx.append(start)
        e_idx.append(end)
        if r is not None:
            r_idx.append(r)

    df = pd.DataFrame(rows)
    ibi = df["contraction_peak_time_s"].diff()
    df["inter_beat_interval_s"] = ibi.where(ibi.notna(), None)
    df["instantaneous_bpm"] = (60.0 / ibi).where(ibi.notna(), None)
    return {
        "cycles_df": df,
        "contraction_idx": np.array(c_idx, dtype=int),
        "relaxation_idx": np.array(r_idx, dtype=int),
        "start_idx": np.array(s_idx, dtype=int),
        "end_idx": np.array(e_idx, dtype=int),
        "baseline_speed": lo,
        "wave_threshold": threshold,
    }
