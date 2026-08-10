import numpy as np
import pytest

from app.analysis.colocalization import analyze_colocalization


def _blob_image(size, cx, cy, r, value=200):
    img = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    img[mask] = value
    return img


def test_identical_channels_perfectly_colocalize():
    # A clean blob on a zero background, rather than full-range noise, so
    # Otsu thresholding cleanly separates "signal" from "background" and
    # M1/M2 = 1 is actually the correct expected value for A == B (with
    # noisy/full-range data, a channel's own below- vs above-threshold
    # split still contributes positive intensity mass either way, so M1
    # legitimately comes out below 1 even when A == B exactly).
    a = _blob_image(100, 50, 50, 25)
    b = a.copy()
    result = analyze_colocalization(a, b)
    assert result["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert result["manders_m1"] == pytest.approx(1.0, abs=1e-6)
    assert result["manders_m2"] == pytest.approx(1.0, abs=1e-6)
    assert result["manders_overlap_coefficient"] == pytest.approx(1.0, abs=1e-6)


def test_completely_separate_blobs_have_low_manders():
    # Channel A only lit up in the top-left, channel B only in the
    # bottom-right — no spatial overlap at all.
    a = _blob_image(100, 20, 20, 10)
    b = _blob_image(100, 80, 80, 10)
    result = analyze_colocalization(a, b)
    assert result["manders_m1"] == pytest.approx(0.0, abs=1e-6)
    assert result["manders_m2"] == pytest.approx(0.0, abs=1e-6)
    assert result["fraction_both_positive"] == pytest.approx(0.0, abs=1e-6)


def test_fully_overlapping_blobs_have_high_manders():
    a = _blob_image(100, 50, 50, 20)
    b = _blob_image(100, 50, 50, 20)
    result = analyze_colocalization(a, b)
    assert result["manders_m1"] == pytest.approx(1.0, abs=1e-6)
    assert result["manders_m2"] == pytest.approx(1.0, abs=1e-6)


def test_partial_overlap_gives_intermediate_manders():
    # Two blobs offset so only part of each overlaps the other.
    a = _blob_image(100, 40, 50, 20)
    b = _blob_image(100, 60, 50, 20)
    result = analyze_colocalization(a, b)
    assert 0.05 < result["manders_m1"] < 0.95
    assert 0.05 < result["manders_m2"] < 0.95


def test_shape_mismatch_raises():
    a = np.zeros((100, 100), dtype=np.uint8)
    b = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError):
        analyze_colocalization(a, b)
