import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
np = pytest.importorskip("numpy")
from experiments.identity.enroll import build_calibrated_gallery


def test_enroll_calibrates_only_unknown_rows():
    rows = [{"sample_id": "a", "cat_id": "cat-a", "split": "gallery"}, {"sample_id": "b", "cat_id": "cat-b", "split": "gallery"}, {"sample_id": "u1", "cat_id": "stranger-1", "split": "calibration_unknown"}, {"sample_id": "u2", "cat_id": "stranger-2", "split": "calibration_unknown"}]
    emb = {"a": np.array([1., 0.]), "b": np.array([0., 1.]), "u1": np.array([.9, .1]), "u2": np.array([.2, .8])}
    result = build_calibrated_gallery(rows, emb, {"model_sha256": "x", "input_size": 160, "embedding_dim": 2}, .01)
    assert result["cat_ids"] == ["cat-a", "cat-b"]
    assert result["n_calibration_unknown"] == 2
    assert result["threshold"] > .9


def test_enroll_requires_unknown_calibration():
    rows = [{"sample_id": "a", "cat_id": "cat-a", "split": "gallery"}]
    with pytest.raises(ValueError, match="calibration_unknown"):
        build_calibrated_gallery(rows, {"a": np.array([1., 0.])}, {"model_sha256": "x", "input_size": 160, "embedding_dim": 2})


def test_enroll_rejects_identity_overlap():
    rows = [{"sample_id": "a", "cat_id": "cat-a", "split": "gallery"}, {"sample_id": "u", "cat_id": "cat-a", "split": "calibration_unknown"}]
    with pytest.raises(ValueError, match="disjoint"):
        build_calibrated_gallery(rows, {"a": np.array([1., 0.]), "u": np.array([0., 1.])}, {"model_sha256": "x", "input_size": 160, "embedding_dim": 2})


def test_enroll_rejects_embedding_dimension_mismatch():
    rows = [{"sample_id": "a", "cat_id": "cat-a", "split": "gallery"},
            {"sample_id": "u", "cat_id": "stranger-1", "split": "calibration_unknown"}]
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        build_calibrated_gallery(rows, {"a": np.array([1., 0.]), "u": np.array([0., 1.])},
                                 {"model_sha256": "x", "input_size": 160, "embedding_dim": 3})
