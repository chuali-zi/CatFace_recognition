import json

import pytest
from PIL import Image

from catid.data import clean_duplicates, load_samples, save_samples, summarize


def sample(path, cat_id="reid:a", task="identity"):
    return dict(path=path, source="reid" if task == "identity" else "catface",
                task=task, cat_id=cat_id, session_id=None, review_status="pending", landmarks=None)


def test_clean_removes_conflicts_and_redundant_same_identity_samples(tmp_path):
    rows = []
    for name, color, identity in [("a.png", "red", "a"), ("b.bmp", "red", "b"),
                                  ("c.png", "blue", "c"), ("d.bmp", "blue", "c")]:
        Image.new("RGB", (8, 8), color).save(tmp_path / name)
        rows.append(sample(name, identity))
    rows.append(sample("missing.png", "d"))
    rows.append(sample("points.png", None, "landmarks"))
    clean, report = clean_duplicates(rows, tmp_path, {"reid"})
    assert [row["path"] for row in clean] == ["c.png", "points.png"]
    assert report["removed_counts"] == {"unreadable": 1, "conflicting_identity": 2, "duplicate": 1}
    assert len(list(tmp_path.iterdir())) == 4


def test_portable_manifest_does_not_count_landmark_groups_as_identities(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "cat.png")
    rows = [sample("cat.png", None, "landmarks")]
    path = tmp_path / "samples.jsonl"
    save_samples(path, rows)
    assert load_samples(path) == rows
    result = summarize(load_samples(path), path.parent)
    assert result["missing_files"] == 0
    assert result["sources"]["catface"]["identities"] == 0


@pytest.mark.parametrize("row", [sample("../outside.png"), sample("cat.png", "catface:directory", "landmarks")])
def test_invalid_manifest(row, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        load_samples(path)
