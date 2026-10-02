import numpy as np
import pytest
from catid.evaluation import evaluate

from experiments.detection.crop_study import evaluate_episode


def _episode():
    def row(sample, cat): return {"sample_id": sample, "cat_id": cat}
    return {"episode_id": "e", "split": {
        "gallery": [row("ga", "a"), row("gb", "b")],
        "known": [row("ka", "a"), row("kb", "b"), row("bad", "a")],
        "calibration_unknown": [row("cu", "c")], "unknown": [row("uu", "u"), row("uf", "u")],
    }}


def _records(include_bad=True, calibration=True):
    values = {"ga": [1, 0], "gb": [0, 1], "ka": [1, 0], "kb": [0, 1],
              "cu": [1, 1], "uu": [1, 0], "uf": [0, 1]}
    if include_bad: values["bad"] = [1, 0]
    result = {key: {"legacy": np.asarray(value, dtype=float), "detected": np.asarray(value, dtype=float),
                    "available": key != "bad" and (calibration or key != "cu"),
                    "detection": {"status": "unusable" if key == "bad" or (key == "cu" and not calibration) else "ok"}}
              for key, value in values.items()}
    return result


def test_failed_enrollment_is_not_silently_dropped():
    records = _records(); records["gb"]["available"] = False
    result = evaluate_episode(_episode(), records, "detected")
    assert result["status"] == "enrollment_incomplete"
    assert result["metrics"] is None


def test_failed_query_stays_in_full_denominator_and_conditional_metric():
    result = evaluate_episode(_episode(), _records(), "detected")
    assert result["status"] == "ok"
    assert result["metrics"]["n_known"] == 3
    assert result["metrics"]["preprocessing_failure"] == pytest.approx(1 / 3)
    assert result["metrics"]["conditional_dir"] == pytest.approx(1.0)


def test_calibration_failure_is_explicit():
    records = _records(calibration=False)
    result = evaluate_episode(_episode(), records, "detected")
    assert result["status"] == "calibration_unavailable"
    assert result["metrics"] is None


def test_legacy_available_rows_are_evaluator_compatible():
    result = evaluate_episode(_episode(), _records(), "legacy")
    assert result["status"] == "ok"
    assert all(row["status"] == "ok" for group in result["predictions"].values() for row in group)
    assert result["metrics"]["n_known"] == 3


def test_all_available_metrics_match_official_evaluator():
    episode = _episode()
    # Use only available rows in this cross-check; this is the evaluator's
    # complete-input contract, while crop-study separately reports failures.
    episode["split"]["known"] = episode["split"]["known"][:2]
    episode["split"]["unknown"] = episode["split"]["unknown"][:1]
    result = evaluate_episode(episode, _records(), "legacy")
    rows = []
    for group, values in result["predictions"].items():
        rows.extend({k: v for k, v in row.items() if k in {"sample_id", "cat_id", "predicted_id", "score"}} | {"split": group}
                    for row in values)
    official = evaluate(rows)
    for field in ("dir", "known_misidentification_rate", "known_reject_rate", "fpir"):
        ours = {"dir": result["metrics"]["dir"], "known_misidentification_rate": result["metrics"]["known_misidentification_rate"],
                "known_reject_rate": result["metrics"]["identity_reject"], "fpir": result["metrics"]["fpir"]}[field]
        assert ours == pytest.approx(official[field])
