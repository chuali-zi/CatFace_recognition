import json

from PIL import Image, ImageOps

from catid.data import save_samples, load_samples
from catid.prepare import HashIndex, prepare, scan


def row(path, identity="individuals:a", task="identity", source="individuals", points=None):
    return dict(path=path, cat_id=identity, task=task, source=source,
                review_status="pending", session_id=None, landmarks=points)


def test_hash_index_matches_brute_force():
    values = [0, 1, 3, 15, 255, 65535, 0xABCDEF1234567890]
    index = HashIndex()
    for number, value in enumerate(values):
        index.add(value, number)
    for query in values + [7, 65432]:
        assert sorted(index.find(query, 4)) == sorted(
            (i, (query ^ value).bit_count()) for i, value in enumerate(values)
            if (query ^ value).bit_count() <= 4)


def test_scan_keeps_outside_landmarks_and_marks_them(tmp_path):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "cat.png")
    points = [[10, 10]] * 8 + [[-2, 80]]
    result = scan(row("cat.png", None, "landmarks", "catface", points), tmp_path)
    assert result["landmarks"] == points
    assert result["landmark_in_frame"] == [True] * 8 + [False]


def test_prepare_cross_source_conflict_and_landmark_dedup(tmp_path):
    image = Image.effect_noise((64, 64), 60).convert("RGB")
    for name in ("a.png", "b.png", "c.png", "d.png"):
        image.save(tmp_path / name)
    points = [[20, 20]] * 9
    rows = [row("a.png"), row("b.png", "hsc:b", source="hsc"),
            row("c.png", None, "landmarks", "catface", points),
            row("d.png", None, "landmarks", "catface", points), row("missing.png")]
    manifest = tmp_path / "samples.jsonl"
    save_samples(manifest, rows)
    report = prepare(manifest)
    assert report["outputs"]["conflicts"] == 2
    assert report["outputs"]["duplicates"] == 1
    assert report["outputs"]["unreadable"] == 1
    assert report["outputs"]["landmarks"] == 1
    assert load_samples(manifest) == rows
    assert load_samples(tmp_path / "prepared.landmarks.jsonl")[0]["review_status"] == "pending"
    assert len(list(tmp_path.glob("*.png"))) == 4


def test_mirror_candidates_stay_out_of_ready_subset_and_cache_tracks_changes(tmp_path):
    image = Image.effect_noise((64, 64), 60).convert("RGB")
    image.save(tmp_path / "a.png")
    ImageOps.mirror(image).save(tmp_path / "b.png")
    manifest = tmp_path / "samples.jsonl"
    save_samples(manifest, [row("a.png"), row("b.png")])
    report = prepare(manifest, radius=0)
    assert report["outputs"]["near-pairs"] == 1
    assert report["outputs"]["identity"] == 0
    assert report["outputs"]["review"] == 2
    assert prepare(manifest, radius=0) == report
    # A changed source must not reuse its old cached pixel fingerprint.
    Image.new("RGB", (65, 65), "red").save(tmp_path / "b.png")
    prepare(manifest, radius=0)
    audit = load_samples(tmp_path / "prepared.audit.jsonl")
    assert audit[1]["width"] == 65


def test_manual_review_applies_to_identical_pixel_copies(tmp_path):
    image = Image.effect_noise((64, 64), 60).convert("RGB")
    image.save(tmp_path / "a.png")
    image.save(tmp_path / "b.png")
    manifest = tmp_path / "samples.jsonl"
    save_samples(manifest, [row("a.png"), row("b.png")])
    (tmp_path / "manual-review.csv").write_text("path,reason\nb.png,visual_multiple_cats_target_ambiguous\n", encoding="utf-8")
    report = prepare(manifest)
    assert report["outputs"]["review"] == 1
    assert report["outputs"]["duplicates"] == 1
    review = load_samples(tmp_path / "prepared.review.jsonl")
    assert review[0]["path"] == "a.png"
    assert "visual_multiple_cats_target_ambiguous" in review[0]["preparation_flags"]


def test_group_review_quarantines_all_identity_images(tmp_path):
    for name in ('a.png', 'b.png'):
        Image.effect_noise((64, 64), 60).convert('RGB').save(tmp_path / name)
    manifest = tmp_path / 'samples.jsonl'
    save_samples(manifest, [row('a.png'), row('b.png')])
    (tmp_path / 'manual-group-review.csv').write_text(
        'cat_id,reason\nindividuals:a,target_unresolved\n', encoding='utf-8-sig')
    result = prepare(manifest, radius=0)
    assert result['outputs']['identity'] == 0
    review = load_samples(tmp_path / 'prepared.review.jsonl')
    assert len(review) == 2
    assert all('target_unresolved' in r['preparation_flags'] for r in review)
