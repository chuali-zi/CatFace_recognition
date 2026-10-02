import json

import numpy as np
import pytest

from experiments.identity.fresh_study import _adapt, _load_rows


def test_protocol_requires_exact_included_images_and_labels(tmp_path):
    row = {"sample_id": "a", "path": "a.jpg", "cat_id": "cat-a", "evaluation_include": True}
    excluded = {**row, "sample_id": "b", "evaluation_include": False}
    (tmp_path / "reviewed.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(excluded), encoding="utf-8")
    path = tmp_path / "protocol.json"
    protocol = {"episodes": [{"split": {"known": [row]}}]}
    path.write_text(json.dumps(protocol), encoding="utf-8")
    assert len(_load_rows(tmp_path)[0]) == 2
    protocol["episodes"][0]["split"]["known"].append(excluded)
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation_include sample set"):
        _load_rows(tmp_path)
    protocol["episodes"][0]["split"]["known"] = [{**row, "cat_id": "wrong"}]
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        _load_rows(tmp_path)


def test_adaptation_preserves_detection_failure_without_normalizing_zero():
    records = {"ok": {"available": True, "detected": np.array([1., 0.])},
               "miss": {"available": False, "detected": np.zeros(2), "detection": {"status": "no_cat"}}}
    result = _adapt(records, {"mean": np.array([0., 0.1]), "projection": None})
    assert result["miss"]["detection"]["status"] == "no_cat"
    assert not result["miss"]["available"]
    assert np.array_equal(result["miss"]["detected"], np.zeros(2))
    assert np.isclose(np.linalg.norm(result["ok"]["detected"]), 1)
    assert np.array_equal(records["ok"]["detected"], [1., 0.])
