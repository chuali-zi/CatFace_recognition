import pytest

from experiments.identity.episodes import build_episode, build_paired_suite, build_suite


def _protocol(cats=8, images=4):
    rows = []
    for cat in range(cats):
        for image in range(images):
            rows.append({"sample_id": f"s-{cat}-{image}", "path": f"p-{cat}-{image}.jpg", "cat_id": f"cat-{cat}"})
    return {
        "train": [{"sample_id": "train", "path": "train.jpg", "cat_id": "train-cat"}],
        "validation": [{"sample_id": "val", "path": "val.jpg", "cat_id": "val-cat"}],
        "development": {"gallery": rows[:2], "known": rows[2:5], "calibration_unknown": rows[5:7], "unknown": rows[7:]},
        "test": {"gallery": [{"sample_id": "t", "path": "t.jpg", "cat_id": "test-cat"}], "known": [], "calibration_unknown": [], "unknown": []},
    }


def test_episode_is_deterministic_complete_and_isolated():
    protocol = _protocol()
    first = build_episode(protocol, gallery_size=2, enrollment=2, seed=17)
    assert first == build_episode(protocol, gallery_size=2, enrollment=2, seed=17)
    rows = [row for values in first.values() for row in values]
    assert len(rows) == 8 * 4
    assert len({row["sample_id"] for row in rows}) == len(rows)
    assert not {row["cat_id"] for row in rows} & {"train-cat", "val-cat", "test-cat"}
    assert {row["cat_id"] for row in first["gallery"]} == {row["cat_id"] for row in first["known"]}
    assert not ({row["cat_id"] for row in first["gallery"]} & {row["cat_id"] for row in first["calibration_unknown"]})


def test_suite_matrix():
    suite = build_suite(_protocol(), seeds=[0, 1], gallery_sizes=[1, 2], enrollments=[2, 3])
    assert len(suite) == 8
    assert {(entry["seed"], entry["gallery_size"], entry["enrollment"]) for entry in suite} == {
        (seed, gallery, enrollment) for seed in [0, 1] for gallery in [1, 2] for enrollment in [2, 3]
    }


def test_paired_suite_keeps_registration_and_queries_fixed():
    protocol = _protocol(images=6)
    suite = build_paired_suite(protocol, seeds=[4], gallery_sizes=[2, 5], enrollments=[3, 5])
    assert len(suite) == 4
    for gallery_size in [2, 5]:
        pair = [entry for entry in suite if entry["gallery_size"] == gallery_size]
        e3, e5 = sorted(pair, key=lambda entry: entry["enrollment"])
        assert e3["episode_id"].startswith("development-paired-")
        assert {row["cat_id"] for row in e3["split"]["gallery"]} == {row["cat_id"] for row in e5["split"]["gallery"]}
        assert e3["split"]["known"] == e5["split"]["known"]
        assert e3["split"]["calibration_unknown"] == e5["split"]["calibration_unknown"]
        assert e3["split"]["unknown"] == e5["split"]["unknown"]
        for cat in {row["cat_id"] for row in e5["split"]["gallery"]}:
            e3_ids = {row["sample_id"] for row in e3["split"]["gallery"] if row["cat_id"] == cat}
            e5_ids = {row["sample_id"] for row in e5["split"]["gallery"] if row["cat_id"] == cat}
            assert e3_ids < e5_ids
        assert not set(e3["unused_enrollment_sample_ids"]) & {
            row["sample_id"] for rows in e3["split"].values() for row in rows
        }
        assert e5["unused_enrollment_sample_ids"] == []
        all_ids = {row["sample_id"] for rows in e5["split"].values() for row in rows}
        all_ids.update(e3["unused_enrollment_sample_ids"])
        assert len(all_ids) == 8 * 6


def test_training_overlap_duplicate_and_insufficient_are_errors():
    protocol = _protocol()
    protocol["train"][0]["cat_id"] = "cat-0"
    with pytest.raises(ValueError, match="overlap"):
        build_episode(protocol, gallery_size=1, enrollment=2)
    protocol = _protocol(cats=3, images=4)
    protocol["development"]["known"][0]["path"] = protocol["development"]["gallery"][0]["path"]
    with pytest.raises(ValueError, match="duplicate path"):
        build_episode(protocol, gallery_size=1, enrollment=2)
    with pytest.raises(ValueError, match="eligible"):
        build_episode(_protocol(cats=2, images=2), gallery_size=1, enrollment=2)
