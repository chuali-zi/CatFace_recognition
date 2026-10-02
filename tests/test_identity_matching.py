import json

import numpy as np
import pytest

from experiments.identity.matching import (build_gallery, predictions,
                                            rank_embedding,
                                            validate_gallery_structure)
from experiments.identity.runtime import make_gallery, match_gallery


def _rows():
    return [{"cat_id": "a", "sample_id": "a1"},
            {"cat_id": "a", "sample_id": "a2"},
            {"cat_id": "b", "sample_id": "b1"}]


def test_prototype_matches_old_runtime_float64_and_json_roundtrip():
    embeddings = {"a1": np.array([1., 2.], dtype=np.float32),
                  "a2": np.array([1., 3.], dtype=np.float32),
                  "b1": np.array([-1., 2.], dtype=np.float32),
                  "q": np.array([.8, 2.1], dtype=np.float32)}
    gallery = build_gallery(_rows(), embeddings)
    ids, vectors = make_gallery(_rows(), embeddings)
    assert gallery["cat_ids"] == ids
    assert np.array_equal(gallery["vectors"], np.asarray(vectors))
    old = match_gallery(embeddings["q"], {"cat_ids": ids, "vectors": vectors})
    result = rank_embedding(embeddings["q"], gallery)
    assert (result["predicted_id"], result["score"]) == old
    restored = json.loads(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                                      for k, v in gallery.items()}))
    assert rank_embedding(embeddings["q"], restored) == result


def test_576d_prototype_matches_legacy_scores_exactly_after_json_roundtrip():
    rng = np.random.default_rng(19)
    rows = [{"cat_id": "cat-a", "sample_id": "a1"},
            {"cat_id": "cat-a", "sample_id": "a2"},
            {"cat_id": "cat-b", "sample_id": "b1"},
            {"cat_id": "cat-c", "sample_id": "c1"}]
    embeddings = {row["sample_id"]: rng.normal(size=576).astype(np.float32)
                  for row in rows}
    query = rng.normal(size=576).astype(np.float32)
    legacy_ids, legacy_vectors = make_gallery(rows, embeddings)
    gallery = build_gallery(rows, embeddings)
    restored = json.loads(json.dumps(gallery))
    expected = match_gallery(query, {"cat_ids": legacy_ids, "vectors": legacy_vectors})
    actual = rank_embedding(query, restored)
    assert actual["predicted_id"] == expected[0]
    assert actual["score"] == expected[1]
    assert actual["similarity"] == expected[1]


def test_max_uses_best_sample_per_identity_and_gap_uses_second_identity():
    rows = [{"cat_id": "a", "sample_id": "a_bad"},
            {"cat_id": "a", "sample_id": "a_best"},
            {"cat_id": "b", "sample_id": "b1"},
            {"cat_id": "b", "sample_id": "b2"}]
    embeddings = {"a_bad": np.array([0., 1.]), "a_best": np.array([1., 0.]),
                  "b1": np.array([.8, .6]), "b2": np.array([-1., 0.]),
                  "q": np.array([1., 0.])}
    gallery = build_gallery(rows, embeddings, aggregation="max", score_mode="gap")
    result = rank_embedding(embeddings["q"], gallery)
    assert result["predicted_id"] == "a"
    assert result["similarity"] == pytest.approx(1.0)
    assert result["runner_up_similarity"] == pytest.approx(.8)
    assert result["score"] == pytest.approx(.2)


def test_single_identity_cosine_allowed_gap_rejected():
    rows = [{"cat_id": "a", "sample_id": "a1"}]
    embeddings = {"a1": np.array([1., 0.]), "q": np.array([1., 0.])}
    gallery = build_gallery(rows, embeddings)
    assert rank_embedding(embeddings["q"], gallery)["runner_up_similarity"] is None
    with pytest.raises(ValueError, match="at least two identities"):
        build_gallery(rows, embeddings, score_mode="gap")


def test_tie_uses_lexicographically_last_identity():
    embeddings = {"a1": np.array([1., 0.]), "b1": np.array([1., 0.]),
                  "q": np.array([1., 0.])}
    result = rank_embedding(embeddings["q"], build_gallery([
        {"cat_id": "a", "sample_id": "a1"}, {"cat_id": "b", "sample_id": "b1"}], embeddings))
    assert result["predicted_id"] == "b"


def test_predictions_include_similarity_observations():
    embeddings = {"a1": np.array([1., 0.]), "b1": np.array([0., 1.]),
                  "q": np.array([1., 0.])}
    rows = {"gallery": [{"cat_id": "a", "sample_id": "a1"},
                         {"cat_id": "b", "sample_id": "b1"}],
            "known": [{"cat_id": "a", "sample_id": "q"}]}
    result = predictions(rows, embeddings)
    assert result[0]["similarity"] == pytest.approx(1)
    assert result[0]["runner_up_similarity"] == pytest.approx(0)


def test_validate_rejects_duplicate_ids_and_bad_groups():
    with pytest.raises(ValueError, match="unique"):
        validate_gallery_structure({"cat_ids": ["a", "a"],
                                    "vectors": [[1.] ]}, embedding_dim=1)
    with pytest.raises(ValueError, match="vector_cat_ids"):
        validate_gallery_structure({"aggregation": "max", "cat_ids": ["a"],
                                    "vectors": [[1.]]}, embedding_dim=1)


@pytest.mark.parametrize("bad_id", [None, "", 3, ["a"]])
def test_build_rejects_non_string_cat_ids(bad_id):
    with pytest.raises(ValueError, match="cat_id"):
        build_gallery([{"cat_id": bad_id, "sample_id": "a1"}],
                      {"a1": np.array([1.])})
