"""Small, fixed feature-space transforms for identity diagnostics."""
from __future__ import annotations

import itertools

import numpy as np


def _unit(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("vectors must be a finite 1-D or 2-D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("vectors must have nonzero norms")
    return x / norms


def fit_transform(vectors, labels, dimensions=128, shrinkage=0.1):
    """Fit a mean-centering plus supervised-whitening/PCA transform.

    Class-pair covariance is the mean outer product of all unordered within-
    identity differences.  Matrix fields in the result are float64 arrays.
    """
    x = _unit(vectors)
    labels = list(labels)
    n, d = x.shape
    if len(labels) != n:
        raise ValueError("labels length must equal number of vectors")
    if len(set(labels)) < 2:
        raise ValueError("at least two identities are required")
    if isinstance(dimensions, bool) or not isinstance(dimensions, (int, np.integer)) or dimensions < 1:
        raise ValueError("dimensions must be a positive integer")
    if dimensions > min(d, n - 1):
        raise ValueError("dimensions must not exceed min(input_dim, n_samples - 1)")
    if not np.isfinite(shrinkage) or not 0 < float(shrinkage) <= 1:
        raise ValueError("shrinkage must be in (0, 1]")
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    diffs = []
    for indices in groups.values():
        for i, j in itertools.combinations(indices, 2):
            diffs.append(x[i] - x[j])
    if not diffs:
        raise ValueError("at least one within-identity pair is required")
    diff = np.asarray(diffs, dtype=np.float64)
    covariance = diff.T @ diff / len(diff)
    trace = float(np.trace(covariance))
    if not np.isfinite(trace) or trace <= 1e-15:
        raise ValueError("within-identity covariance has no variation")
    regularized = (1.0 - float(shrinkage)) * covariance + float(shrinkage) * (trace / d) * np.eye(d)
    values, vectors_eig = np.linalg.eigh((regularized + regularized.T) * 0.5)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("regularized covariance is not positive definite")
    whitening = (vectors_eig * (1.0 / np.sqrt(values))) @ vectors_eig.T
    centered = x - np.mean(x, axis=0)
    whitened = centered @ whitening
    pca_cov = whitened.T @ whitened / n
    pca_values, pca_vectors = np.linalg.eigh((pca_cov + pca_cov.T) * 0.5)
    order = np.argsort(pca_values)[::-1][:int(dimensions)]
    projection = whitening @ pca_vectors[:, order]
    return {"mean": np.asarray(np.mean(x, axis=0), dtype=np.float64), "projection": np.asarray(projection, dtype=np.float64),
            "input_dim": d, "output_dim": int(dimensions), "n_samples": n,
            "n_identities": len(groups), "n_pairs": len(diff), "shrinkage": float(shrinkage),
            "pair_covariance": np.asarray(covariance, dtype=np.float64),
            "method": "unit64_mean_class_pair_covariance_shrinkage_whiten_pca"}


def apply_transform(vectors, transform):
    """Apply a fitted transform and return float64 unit-norm rows."""
    single = np.asarray(vectors).ndim == 1
    x = _unit(vectors)
    if not isinstance(transform, dict) or "mean" not in transform:
        raise ValueError("invalid transform")
    mean = np.asarray(transform["mean"], dtype=np.float64)
    projection = transform.get("projection")
    if mean.ndim != 1 or mean.size != x.shape[1] or not np.isfinite(mean).all():
        raise ValueError("transform mean has the wrong shape")
    if projection is None:
        output = x - mean
    else:
        projection = np.asarray(projection, dtype=np.float64)
        if projection.ndim != 2 or projection.shape[0] != x.shape[1] or not np.isfinite(projection).all():
            raise ValueError("transform projection has the wrong shape")
        output = (x - mean) @ projection
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
        raise ValueError("transformed vectors have zero or non-finite norm")
    result = output / norms
    return result[0] if single else result
