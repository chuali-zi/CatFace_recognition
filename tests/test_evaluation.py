import math

import pytest

from catid.evaluation import evaluate, threshold_at_fpir


def predictions():
    return [
        dict(sample_id="cal", split="calibration_unknown", cat_id="stranger-a", predicted_id="a", score=0.8),
        dict(sample_id="k1", split="known", cat_id="a", predicted_id="b", score=0.99),
        dict(sample_id="k2", split="known", cat_id="a", predicted_id="a", score=0.95),
        dict(sample_id="k3", split="known", cat_id="b", predicted_id="b", score=0.5),
        dict(sample_id="u1", split="unknown", cat_id="stranger-b", predicted_id="a", score=0.9),
        dict(sample_id="u2", split="unknown", cat_id="stranger-b", predicted_id="b", score=0.6),
    ]


def test_wrong_identity_is_not_success_and_unknown_denominator_is_probes():
    result = evaluate(predictions())
    assert result["top1"] == 2 / 3
    assert result["dir"] == 1 / 3
    assert result["known_misidentification_rate"] == 1 / 3
    assert result["known_reject_rate"] == 1 / 3
    assert result["fpir"] == 0.5
    assert result["n_unknown"] == 2


def test_tied_unknown_scores_are_rejected_together():
    assert threshold_at_fpir([0.9] * 20, 0.01) > 0.9


def test_test_scores_cannot_change_calibration_threshold():
    rows = predictions()
    threshold = evaluate(rows)["threshold"]
    rows[-1]["score"] = 100
    assert evaluate(rows)["threshold"] == threshold


@pytest.mark.parametrize("change", ["overlap", "duplicate", "nan", "missing", "unknown_in_gallery"])
def test_invalid_evaluation_inputs(change):
    rows = predictions()
    if change == "overlap":
        rows[-1]["cat_id"] = "stranger-a"
    elif change == "duplicate":
        rows[-1]["sample_id"] = "cal"
    elif change == "nan":
        rows[-1]["score"] = math.nan
    elif change == "missing":
        rows = rows[1:]
    else:
        rows[1]["predicted_id"] = "stranger-b"
    with pytest.raises(ValueError):
        evaluate(rows)
