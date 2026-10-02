import numpy as np
import pytest

from experiments.identity.metric import apply_transform, fit_transform
from experiments.identity.metric_study import training_rows


def test_low_rank_regularized_transform_is_finite_and_unit():
    x = np.asarray([[1, 0, 0, 0], [0.99, .1, 0, 0], [0, 1, 0, 0], [0, .99, .1, 0]], dtype=float)
    transform = fit_transform(x, ["a", "a", "b", "b"], dimensions=2, shrinkage=.1)
    output = apply_transform(x, transform)
    assert output.shape == (4, 2)
    assert output.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0)
    assert apply_transform(x[0], transform).shape == (2,)


def test_pair_covariance_and_linear_projection_math():
    x = np.asarray([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
    transform = fit_transform(x, ["a", "a", "b", "b"], dimensions=1, shrinkage=1.0)
    assert transform["n_pairs"] == 2
    assert transform["mean"].dtype == np.float64
    assert transform["projection"].dtype == np.float64
    normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    expected = np.outer(normalized[0] - normalized[1], normalized[0] - normalized[1])
    expected += np.outer(normalized[2] - normalized[3], normalized[2] - normalized[3])
    np.testing.assert_allclose(transform["pair_covariance"], expected / 2)
    assert np.asarray(transform["projection"]).shape == (2, 1)
    normalized_row = x[:1] / np.linalg.norm(x[:1], axis=1, keepdims=True)
    manual = normalized_row - transform["mean"]
    manual = manual @ transform["projection"]
    manual /= np.linalg.norm(manual, axis=1, keepdims=True)
    np.testing.assert_allclose(apply_transform(x[:1], transform), manual)
    cs = np.asarray(transform["pair_covariance"])
    cs_reg = transform["shrinkage"] * np.trace(cs) / 2 * np.eye(2) + (1 - transform["shrinkage"]) * cs
    projection = transform["projection"]
    np.testing.assert_allclose(projection.T @ cs_reg @ projection, np.eye(1), atol=1e-10)


def test_rejects_zero_vectors_and_bad_shapes():
    with pytest.raises(ValueError):
        fit_transform([[0, 0], [1, 0], [0, 1]], ["a", "a", "b"], dimensions=1)
    with pytest.raises(ValueError):
        fit_transform([[1, 0], [0, 1]], ["a"], dimensions=1)
    with pytest.raises(ValueError):
        apply_transform([[1, 0]], {"mean": [0, 0], "projection": [[1], [1, 0]]})


def test_metric_fit_rows_are_original_train_only():
    protocol = {"train": [{"sample_id": "train-1", "cat_id": "cat-a"}],
                "development": {"gallery": [{"sample_id": "dev-1", "cat_id": "cat-b"}]},
                "test": {"gallery": []}}
    prepared = {"train": [{"sample_id": "train-1", "cat_id": "cat-a", "crop_box": [0, 0, 2, 2]}],
                "training_crop_pipeline": {"mode": "head"}}
    assert training_rows(protocol, prepared) == prepared["train"]
    prepared["train"] = [{"sample_id": "dev-1", "cat_id": "cat-b", "crop_box": [0, 0, 2, 2]}]
    with pytest.raises(ValueError):
        training_rows(protocol, prepared)
    prepared["train"] = [{"sample_id": "validation-1", "cat_id": "cat-a", "crop_box": [0, 0, 2, 2]}]
    with pytest.raises(ValueError):
        training_rows(protocol, prepared)
