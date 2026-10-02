import json
import pytest
from experiments.detection.enroll import _write_failure, validate_rows


def test_validate_rows_rejects_wrong_split_and_empty_cat():
    with pytest.raises(ValueError, match="split"):
        list(validate_rows([{"path":"a.png","cat_id":"a","split":"query"}]))
    with pytest.raises(ValueError, match="cat_id"):
        list(validate_rows([{"path":"a.png","cat_id":"","split":"gallery"}]))


def test_failure_diagnostic_keeps_all_states_and_calibration_counts(tmp_path):
    rows=[{"path":"a.png","cat_id":"a","split":"gallery"}, {"path":"u.png","cat_id":"u","split":"calibration_unknown"}]
    states=[{"sample_id":"a","status":"failed","error":"detector status: no_cat"}, {"sample_id":"u","status":"failed","error":"duplicate image pixels"}]
    path=_write_failure(tmp_path/"gallery.json", rows, states, "gallery enrollment failed")
    value=json.loads(path.read_text(encoding="utf-8"))
    assert value["n_rows"] == 2 and len(value["states"]) == 2
    assert sum(s["status"] == "failed" for s in value["states"]) == 2


def test_validate_normalizes_sample_id_and_rejects_role_overlap(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"y")
    rows=[{"path":"a.png","cat_id":"cat-a","split":"gallery"},
          {"path":"b.png","cat_id":"cat-a","split":"calibration_unknown"}]
    with pytest.raises(ValueError, match="disjoint"):
        validate_rows(rows, tmp_path)
