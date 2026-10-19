import json
import math

import pytest

from experiments.identity.ambiguity_study import compare_scores, decide, summarize


def ranked(score=.875, runner=.872, **kwargs):
    return {"predicted_id": "cat-a", "score": score, "similarity": score,
            "runner_up_similarity": runner, **kwargs}


def test_default_preserves_threshold_boundary_and_tied_candidates():
    assert decide(ranked(score=.8, runner=.8), .8)["accepted"]
    assert not decide(ranked(score=.799, runner=.79), .8)["accepted"]
    assert decide(ranked(runner=None), .8)["accepted"]


def test_ambiguity_only_removes_accepts_and_explains_rejection():
    result = decide(ranked(), .864, .005)
    assert result["baseline_accepted"]
    assert result["status"] == "unknown" and result["cat_id"] is None
    assert result["reason"] == "ambiguous_identity"
    assert decide(ranked(score=.875, runner=.5), .864, .005)["accepted"]
    assert decide(ranked(score=.8, runner=.7), .864, .005)["reason"] == "below_score_threshold"


def test_margin_boundary_is_inclusive_and_single_identity_is_unconfirmed():
    assert decide(ranked(score=.875, runner=.75), .8, .125)["accepted"]
    assert decide(ranked(runner=None), .8, .005)["reason"] == "insufficient_candidates"


@pytest.mark.parametrize("value", [-.1, math.nan, math.inf, True])
def test_invalid_margins_fail_explicitly(value):
    with pytest.raises(ValueError):
        decide(ranked(), .8, value)


@pytest.mark.parametrize("row", [ranked(score=math.nan), ranked(runner=math.inf),
                                  ranked(runner=.99), ranked(predicted_id="")])
def test_invalid_ranked_data_cannot_become_a_known_identity(row):
    with pytest.raises(ValueError):
        decide(row, .8, .005)


@pytest.mark.parametrize("status", ["no_cat", "unusable", "error"])
def test_preprocessing_failures_are_preserved(status):
    assert decide({"available": False, "status": status}, .8, .005)["status"] == status


def test_failures_remain_in_denominators_and_calibration_is_separate():
    failed = decide({"available": False, "status": "no_cat"}, .8)
    accepted = decide(ranked(), .8)
    rows = [dict(split="known", true_cat_id="cat-a", candidate=failed),
            dict(split="known", true_cat_id="cat-b", candidate=accepted),
            dict(split="unknown", true_cat_id="cat-c", candidate=failed),
            dict(split="calibration_unknown", true_cat_id="cat-d", candidate=accepted)]
    result = summarize(rows, "candidate")
    assert result["known"]["queries"] == 2
    assert result["known"]["wrong_accepts"] == 1
    assert result["unknown"]["preprocessing_failures"] == 1
    assert result["unknown"]["false_accepts"] == 0
    assert result["calibration_unknown"]["false_accepts"] == 1


def test_frozen_comparison_preserves_default_and_reports_added_rejection(tmp_path):
    (tmp_path / "models/large224").mkdir(parents=True)
    (tmp_path / "evidence").mkdir()
    gallery = tmp_path / "models/large224/gallery.json"
    evidence = tmp_path / "evidence/large224-detected.json"
    gallery.write_text(json.dumps({"threshold": .864}))
    rows = {name: [] for name in ["calibration_unknown", "known", "unknown"]}
    rows["known"] = [{**ranked(), "sample_id": "a", "split": "known", "cat_id": "cat-b",
                      "threshold": .864, "accepted": True}]
    evidence.write_text(json.dumps({"predictions": rows}))
    disabled = compare_scores(tmp_path, ["large224"], 0)[0]
    assert disabled["baseline"] == disabled["candidate"]
    assert disabled["changed_queries"] == 0
    enabled = compare_scores(tmp_path, ["large224"], .005)[0]
    assert enabled["baseline"]["known"]["wrong_accepts"] == 1
    assert enabled["candidate"]["known"]["wrong_accepts"] == 0
    assert enabled["candidate"]["known"]["not_accepted"] == 1
    assert enabled["changed_queries"] == 1
    rows["known"][0]["accepted"] = False
    evidence.write_text(json.dumps({"predictions": rows}))
    with pytest.raises(ValueError, match="reproduce"):
        compare_scores(tmp_path, ["large224"], .005)
