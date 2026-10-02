import pytest

from experiments.identity.fresh_data import build_reviewed_episodes, merge_reviews, select_candidates


def test_selection_is_reproducible_and_excludes_all_existing_identity_roles():
    protocol = {"train": [{"cat_id": "a"}], "validation": [{"cat_id": "a"}],
                "development": {"gallery": [{"cat_id": "b"}]}, "test": {"unknown": [{"cat_id": "c"}]}}
    rows = [{"source": "individuals", "cat_id": cat, "path": f"{cat}/{i}.jpg", "pixel_hash": f"{cat}-{i}",
             "dhash": f"{i:016x}", "dhash_mirror": f"{i+1:016x}"}
            for cat in "abcde" for i in range(3)]
    first, meta = select_candidates(rows, protocol, set(), identities=2, per_identity=2)
    second, _ = select_candidates(list(reversed(rows)), protocol, set(), identities=2, per_identity=2)
    assert first == second
    assert {r["cat_id"] for r in first} == {"d", "e"}
    assert len(first) == 4 and meta["new_pool_images"] == 6
    with pytest.raises(ValueError, match="duplicates an old"):
        select_candidates(rows, protocol, {"d-0"}, identities=2, per_identity=2)


def test_selection_rejects_cross_label_duplicate_pixels():
    protocol = {"train": [], "validation": [], "development": {}, "test": {}}
    rows = [{"source": "individuals", "cat_id": cat, "path": f"{cat}.jpg", "pixel_hash": "same",
             "dhash": "1", "dhash_mirror": "2"} for cat in "ab"]
    with pytest.raises(ValueError, match="duplicate pixels"):
        select_candidates(rows, protocol, set(), identities=2, per_identity=1)


def test_review_requires_every_image_and_quarantines_the_whole_identity():
    rows = [{"review_index": i, "cat_id": "a", "sample_id": f"a{i}", "status": "no_cat"}
            for i in range(2)]
    group = {"cat_id": "a", "group_decision": "appearance_consistent", "group_note": "matching visible markings",
             "images": [{"review_index": i, "visual": "difficult", "reason": "visible cat, no box"} for i in range(2)]}
    merged = merge_reviews(rows, [{"groups": [group]}])
    assert all(r["evaluation_include"] and r["status"] == "no_cat" for r in merged)
    isolated = merge_reviews(rows, [{"groups": [group]}], {"a"})
    assert all(not r["evaluation_include"] for r in isolated)
    with pytest.raises(ValueError, match="missing visual"):
        merge_reviews(rows, [])
    with pytest.raises(ValueError, match="indices"):
        merge_reviews(rows, [{"groups": [{**group, "images": group["images"][:1]}]}])
    wrong = {**group, "images": [{**d, "sample_id": "wrong"} for d in group["images"]]}
    with pytest.raises(ValueError, match="sample_id"):
        merge_reviews(rows, [{"groups": [wrong]}])


def test_reviewed_episodes_preserve_difficult_queries_and_isolate_roles():
    rows = [{"cat_id": cat, "sample_id": f"{cat}-{i}", "path": f"{cat}/{i}.jpg",
             "evaluation_include": cat != "excluded", "visual": "usable" if i < 3 else "difficult",
             "status": "ok" if i < 3 else "no_cat"}
            for cat in ["a", "b", "c", "d", "e", "f", "excluded"] for i in range(4)]
    suite, eligible = build_reviewed_episodes(rows, count=3, gallery_size=2, enrollment=3)
    assert (suite, eligible) == build_reviewed_episodes(list(reversed(rows)), count=3, gallery_size=2, enrollment=3)
    assert "excluded" not in eligible
    for episode in suite:
        split = episode["split"]
        all_rows = [r for values in split.values() for r in values]
        assert len(all_rows) == len({r["sample_id"] for r in all_rows}) == 24
        ids = {k: {r["cat_id"] for r in v} for k, v in split.items()}
        assert ids["known"] == ids["gallery"]
        assert not (ids["gallery"] & (ids["unknown"] | ids["calibration_unknown"]))
        assert not (ids["unknown"] & ids["calibration_unknown"])
        assert all(r["sample_id"].endswith("-3") for r in split["known"])
