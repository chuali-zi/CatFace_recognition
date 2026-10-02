"""Small NumPy-only gallery construction and matching helpers."""

from __future__ import annotations

import numpy as np

from .runtime import unit64


_AGGREGATIONS = {"prototype", "max"}
_SCORE_MODES = {"cosine", "gap"}


def _check_options(aggregation: str, score_mode: str) -> None:
    if aggregation not in _AGGREGATIONS:
        raise ValueError("aggregation must be prototype or max")
    if score_mode not in _SCORE_MODES:
        raise ValueError("score_mode must be cosine or gap")


def build_gallery(rows: list[dict], embeddings: dict, aggregation: str = "prototype",
                  score_mode: str = "cosine") -> dict:
    """Build a gallery with vectors as standard JSON-compatible lists."""
    _check_options(aggregation, score_mode)
    if not rows:
        raise ValueError("gallery rows must not be empty")
    if any(not isinstance(row.get("cat_id"), str) or not row["cat_id"] for row in rows):
        raise ValueError("cat_id must be a non-empty string")
    identities = sorted({row["cat_id"] for row in rows})
    if score_mode == "gap" and len(identities) < 2:
        raise ValueError("gap scoring requires at least two identities")
    sample_vectors = []
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in embeddings:
            raise ValueError(f"missing embedding for {sample_id}")
        sample_vectors.append(unit64(embeddings[sample_id]))
    dimension = sample_vectors[0].size
    if any(vector.size != dimension for vector in sample_vectors):
        raise ValueError("embedding dimensions must match")

    gallery = {"aggregation": aggregation, "score_mode": score_mode,
               "cat_ids": identities}
    if aggregation == "prototype":
        vectors = []
        for cat_id in identities:
            values = [vector for row, vector in zip(rows, sample_vectors)
                      if row["cat_id"] == cat_id]
            vectors.append(unit64(np.mean(values, axis=0)))
        gallery["vectors"] = np.asarray(vectors, dtype=np.float64).tolist()
    else:
        gallery["vectors"] = np.asarray(sample_vectors, dtype=np.float64).tolist()
        gallery["vector_cat_ids"] = [row["cat_id"] for row in rows]
    validate_gallery_structure(gallery, embedding_dim=dimension)
    return gallery


def validate_gallery_structure(gallery: dict, embedding_dim: int = 576) -> None:
    """Raise ValueError when a gallery is malformed or incompatible."""
    if not isinstance(gallery, dict):
        raise ValueError("gallery must be a dict")
    aggregation = gallery.get("aggregation", "prototype")
    score_mode = gallery.get("score_mode", "cosine")
    _check_options(aggregation, score_mode)
    cat_ids = gallery.get("cat_ids")
    vectors = np.asarray(gallery.get("vectors"), dtype=np.float64)
    if (not isinstance(cat_ids, list) or not cat_ids or
            any(not isinstance(cat_id, str) or not cat_id for cat_id in cat_ids)):
        raise ValueError("cat_ids must be a non-empty list of strings")
    if len(set(cat_ids)) != len(cat_ids):
        raise ValueError("cat_ids must be a non-empty list of unique identities")
    if score_mode == "gap" and len(cat_ids) < 2:
        raise ValueError("gap scoring requires at least two identities")
    if vectors.ndim != 2 or vectors.shape[1] != embedding_dim:
        raise ValueError("gallery vectors have an invalid shape")
    if not np.isfinite(vectors).all():
        raise ValueError("gallery vectors must be finite")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0) or not np.allclose(norms, 1.0, rtol=0, atol=1e-12):
        raise ValueError("gallery vectors must be unit norm")
    if aggregation == "prototype":
        if vectors.shape[0] != len(cat_ids):
            raise ValueError("prototype vectors must correspond to cat_ids")
    else:
        vector_cat_ids = gallery.get("vector_cat_ids")
        if not isinstance(vector_cat_ids, list) or len(vector_cat_ids) != vectors.shape[0]:
            raise ValueError("max gallery needs one vector_cat_ids entry per vector")
        if set(vector_cat_ids) != set(cat_ids):
            raise ValueError("vector_cat_ids must contain every cat_id exactly by group")


def _identity_scores(probe: np.ndarray, gallery: dict) -> dict:
    vectors = np.asarray(gallery["vectors"], dtype=np.float64)
    if gallery.get("aggregation", "prototype") == "prototype":
        return {cat_id: float(np.dot(probe, vector))
                for cat_id, vector in zip(gallery["cat_ids"], vectors)}
    scores = {cat_id: -np.inf for cat_id in gallery["cat_ids"]}
    for cat_id, vector in zip(gallery["vector_cat_ids"], vectors):
        scores[cat_id] = max(scores[cat_id], float(np.dot(probe, vector)))
    return scores


def rank_embedding(embedding: np.ndarray, gallery: dict) -> dict:
    """Rank a probe and return top/runner-up observations (without a gate)."""
    probe = unit64(embedding)
    validate_gallery_structure(gallery, embedding_dim=probe.size)
    scores = _identity_scores(probe, gallery)
    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    predicted_id, similarity = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else None
    score = similarity if gallery.get("score_mode", "cosine") == "cosine" else similarity - runner_up
    return {"predicted_id": predicted_id, "score": float(score),
            "similarity": float(similarity),
            "runner_up_similarity": None if runner_up is None else float(runner_up)}


def predictions(split: dict, embeddings: dict, aggregation: str = "prototype",
                score_mode: str = "cosine") -> list[dict]:
    """Produce the standard per-query prediction rows plus score observations."""
    gallery = build_gallery(split.get("gallery", []), embeddings, aggregation, score_mode)
    output = []
    for split_name in ("known", "calibration_unknown", "unknown"):
        for row in split.get(split_name, []):
            sample_id = row["sample_id"]
            if sample_id not in embeddings:
                raise ValueError(f"missing embedding for {sample_id}")
            result = rank_embedding(embeddings[sample_id], gallery)
            output.append({"sample_id": sample_id, "split": split_name,
                           "cat_id": row["cat_id"], **result})
    return output
