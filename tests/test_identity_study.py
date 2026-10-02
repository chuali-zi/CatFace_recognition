import json

import pytest

np = pytest.importorskip("numpy")

from experiments.identity import study


def test_summary_uses_episode_macro_not_repeated_image_denominators():
    fields = ("top1", "dir", "fpir", "known_misidentification_rate", "known_reject_rate")
    rows = [{**dict.fromkeys(fields, value), "n_known": count}
            for value, count in ((0.0, 2), (1.0, 100))]
    report = study.summarize(rows)
    assert report["n_episodes"] == 2
    assert report["dir"]["mean"] == 0.5
    assert report["dir"]["min"] == 0.0
    assert report["dir"]["max"] == 1.0


def test_cache_reuses_features_and_invalidates_changed_model_metadata(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test model; not a real ONNX")
    metadata = model.with_suffix(".json")
    metadata.write_text('{"input_size":160,"embedding_dim":8,"architecture":"fake-8"}', encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    (tmp_path / "face.png").write_bytes(b"mock image")
    protocol = {"development": {"gallery": [
        {"sample_id": "q", "cat_id": "cat", "path": "face.png"}]}}
    calls = []

    class FakeEncoder:
        def __init__(self, path, threads):
            self.metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.size = self.metadata["input_size"]

    def fake_embed(encoder, rows, root):
        calls.append(encoder.size)
        return {row["sample_id"]: np.ones(8, dtype=np.float32) for row in rows}

    monkeypatch.setattr(study, "Encoder", FakeEncoder)
    monkeypatch.setattr(study, "embed_rows", fake_embed)
    cache = tmp_path / "cache" / "features.npz"
    args = (model, manifest, protocol, "development", cache, 1)
    first = study.cached_embeddings(*args)
    second = study.cached_embeddings(*args)
    assert calls == [160]
    np.testing.assert_array_equal(first["q"], second["q"])
    assert first["q"].shape == (8,)
    sidecar = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["embedding_dim"] == 8
    np.savez_compressed(cache, sample_ids=np.asarray(["q"]), embeddings=np.ones((1, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="Invalid embedding cache"):
        study.cached_embeddings(*args)
    metadata.write_text('{"input_size":160,"embedding_dim":8,"revision":2}', encoding="utf-8")
    study.cached_embeddings(*args)
    assert calls == [160, 160]
    metadata.write_text('{"input_size":160,"embedding_dim":8,"training_identities":["cat"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        study.cached_embeddings(*args)
