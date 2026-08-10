import numpy as np
import pytest

from app.analysis.validation_stats import compute_agreement


def test_perfect_agreement():
    a = [10.0, 20.0, 30.0, 40.0, 50.0, 25.0, 35.0]
    b = list(a)
    result = compute_agreement(a, b)
    assert result["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    assert result["icc_2_1"] == pytest.approx(1.0, abs=1e-6)
    assert result["bland_altman_bias"] == pytest.approx(0.0, abs=1e-9)
    assert result["bland_altman_sd_diff"] == pytest.approx(0.0, abs=1e-9)


def test_perfect_correlation_but_systematic_bias():
    # b is always a + 5: correlation is perfect, but the methods don't
    # actually agree in absolute terms — ICC and Bland-Altman must catch
    # this even though Pearson r alone would suggest perfect agreement.
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 25.0, 35.0])
    b = a + 5.0
    result = compute_agreement(a.tolist(), b.tolist())
    assert result["pearson_r"] == pytest.approx(1.0, abs=1e-9)
    assert result["bland_altman_bias"] == pytest.approx(-5.0, abs=1e-6)
    assert result["icc_2_1"] < 0.95  # absolute agreement penalized despite r=1


def test_no_agreement_low_correlation_and_icc():
    rng = np.random.default_rng(0)
    a = rng.normal(50, 10, 30)
    b = rng.normal(50, 10, 30)  # independent of a
    result = compute_agreement(a.tolist(), b.tolist())
    assert abs(result["pearson_r"]) < 0.4
    assert result["icc_2_1"] < 0.4


def test_requires_paired_equal_length():
    with pytest.raises(ValueError):
        compute_agreement([1, 2, 3], [1, 2])


def test_requires_minimum_observations():
    with pytest.raises(ValueError):
        compute_agreement([1, 2], [1, 2])


def test_bland_altman_limits_of_agreement_bracket_diffs_for_normal_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 5, 200)
    noise = rng.normal(0, 2, 200)
    b = a + noise
    result = compute_agreement(a.tolist(), b.tolist())
    diffs = np.array(result["diffs"])
    within = np.mean((diffs >= result["bland_altman_loa_lower"]) & (diffs <= result["bland_altman_loa_upper"]))
    # ~95% of differences should fall within the 95% limits of agreement by construction
    assert within > 0.90
