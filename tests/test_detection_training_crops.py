import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from PIL import Image
import pytest
import numpy as np
from experiments.detection.training_crops import prepare_rows


class FakeDetector:
    model_sha256 = "det"
    metadata_sha256 = "meta"
    def detect(self, image):
        return np.asarray([[1, 1, 8, 8]], dtype=np.float32), np.asarray([0.9], dtype=np.float32)


def test_head_and_legacy_keep_identical_rows_and_isolate_cats(tmp_path):
    for name in ("a-train.png", "a-val.png", "b-train.png", "b-val.png"):
        Image.new("RGB", (10, 10), "white").save(tmp_path / name)
    def row(name, cat):
        return {"path": name, "original_path": name, "cat_id": cat, "face_box": [2, 2, 6, 6]}
    manifest = [row("a-train.png", "a"), row("a-val.png", "a"), row("b-train.png", "b")]
    protocol = {"train": [{"sample_id": "a-train.png", "path": "a-train.png", "cat_id": "a"}, {"sample_id": "b-train.png", "path": "b-train.png", "cat_id": "b"}], "validation": [{"sample_id": "a-val.png", "path": "a-val.png", "cat_id": "a"}], "development": {}, "test": {}}
    records, head, legacy = prepare_rows(manifest, protocol, tmp_path, FakeDetector())
    assert [(r["sample_id"], r["cat_id"]) for r in head] == [("a-train.png", "a"), ("a-val.png", "a")]
    assert [(r["sample_id"], r["cat_id"]) for r in head] == [(r["sample_id"], r["cat_id"]) for r in legacy]
    assert all("crop_box" in r for r in head)
    protocol["development"] = {"known": [{"cat_id": "a", "path": "not-opened.png"}]}
    with pytest.raises(ValueError, match="overlap"):
        prepare_rows(manifest, protocol, tmp_path, FakeDetector())


def test_failed_validation_is_not_filled_or_kept(tmp_path):
    Image.new("RGB", (10, 10), "white").save(tmp_path / "a.png")
    class NoDetector(FakeDetector):
        def detect(self, image): return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    Image.new("RGB", (10, 10), "white").save(tmp_path / "a-val.png")
    manifest = [{"path": "a.png", "original_path": "a.png", "cat_id": "a", "face_box": [2, 2, 6, 6]}, {"path": "a-val.png", "original_path": "a-val.png", "cat_id": "a", "face_box": [2, 2, 6, 6]}]
    protocol = {"train": [{"sample_id": "a.png", "path": "a.png", "cat_id": "a"}], "validation": [{"sample_id": "a-val.png", "path": "a-val.png", "cat_id": "a"}], "development": {}, "test": {}}
    records, head, legacy = prepare_rows(manifest, protocol, tmp_path, NoDetector())
    assert head == legacy == []
    assert len(records) == 2 and all(r["status"] == "no_cat" for r in records)
