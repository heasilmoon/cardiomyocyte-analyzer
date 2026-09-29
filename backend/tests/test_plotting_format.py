from app.analysis.plotting import _format_p, _metric_display


def test_format_p_nejm_style_matches_prism():
    assert _format_p(0.12) == "p = 0.12 (ns)"
    assert _format_p(0.033) == "p = 0.033 (*)"
    assert _format_p(0.002) == "p = 0.002 (**)"
    assert _format_p(0.0004) == "p < 0.001 (***)"
    assert _format_p(0.123456, "nejm") == "p = 0.12 (ns)"
    assert _format_p(None) == "p = n/a"


def test_format_p_other_styles():
    assert _format_p(0.0123, "value") == "p = 0.0123"
    assert _format_p(0.00001, "value") == "p < 0.0001"
    assert _format_p(0.00001, "stars") == "****"
    assert _format_p(0.03, "stars") == "*"
    assert _format_p(0.5, "stars") == "ns"


def test_metric_display_titles_and_units():
    assert _metric_display("mean_bpm") == ("Beating rate", "BPM")
    assert _metric_display("mean_max_contraction_speed", {"units": {"speed_units": "um/s"}}) == (
        "Max. contraction speed (MCS)", "µm/s"
    )
    # Unknown keys are prettified and their unit suffix recognised.
    assert _metric_display("mean_some_new_thing_s") == ("Some new thing", "s")
