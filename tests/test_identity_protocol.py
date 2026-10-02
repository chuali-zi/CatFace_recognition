import json

import pytest

np = pytest.importorskip("numpy")

from experiments.identity.protocol import (
    build_protocol,
    evaluate_embeddings,
    predictions_from_embeddings,
)


def _manifest(path, cats=6, images=4):
    with path.open("w", encoding="utf-8") as stream:
        for cat_index in range(cats):
            for image_index in range(images):
                sample = f"images/{cat_index}-{image_index}.jpg"
                stream.write(json.dumps({"path": sample, "cat_id": f"cat-{cat_index}",
                                         "pixel_hash": f"p-{cat_index}-{image_index}",
                                         "original_pixel_hash": f"o-{cat_index}-{image_index}",
                                         "session_id": "none"}) + "\n")


def test_protocol_is_reproducible_and_isolates_identities(tmp_path):
    manifest = tmp_path / "clean.identity.jsonl"
    _manifest(manifest, cats=15)
    first = build_protocol(manifest, gallery_size=1, enrollment=2)
    second = build_protocol(manifest, gallery_size=1, enrollment=2)
    assert first == second
    groups = {part: {row["cat_id"] for row in first[part]}
              for part in ("train", "validation")}
    groups["development"] = {row["cat_id"] for rows in first["development"].values() for row in rows}
    groups["test"] = {row["cat_id"] for rows in first["test"].values() for row in rows}
    assert not (groups["train"] | groups["validation"]) & groups["development"]
    assert not groups["development"] & groups["test"]
    assert first["diagnostic_only"] is True


def test_duplicate_hash_and_insufficient_data_are_rejected(tmp_path):
    manifest = tmp_path / "clean.identity.jsonl"
    _manifest(manifest, cats=6, images=4)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(lines[-1]); duplicate["path"] = "images/duplicate.jpg"
    lines.append(json.dumps(duplicate)); manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        build_protocol(manifest, gallery_size=1, enrollment=2)

    _manifest(manifest, cats=2, images=4)
    with pytest.raises(ValueError, match="three identities"):
        build_protocol(manifest, gallery_size=1, enrollment=2)


def test_embedding_predictions_and_errors(tmp_path):
    manifest = tmp_path / "clean.identity.jsonl"
    _manifest(manifest, cats=15)
    protocol = build_protocol(manifest, gallery_size=1, enrollment=2)
    split = protocol["development"]
    embeddings = {}
    for rows in split.values():
        for row in rows:
            vector = np.zeros(4, dtype=float)
            vector[int(row["cat_id"].split("-")[1]) % 4] = 1
            embeddings[row["sample_id"]] = vector
    predictions, metrics = evaluate_embeddings(protocol, embeddings)
    assert predictions and "top1" in metrics
    with pytest.raises(ValueError, match="missing embedding"):
        predictions_from_embeddings(split, {})
    with pytest.raises(ValueError, match="only prototype"):
        predictions_from_embeddings(split, embeddings, aggregation="max")
