"""Compare legacy and detector crops using precomputed identity features.

This is an image-level diagnostic: episodes reuse the existing development
protocol and are not independent cats or visits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from catid.evaluation import threshold_at_fpir
from experiments.identity.episodes import build_paired_suite
from experiments.identity.matching import build_gallery, rank_embedding


MODES = ("legacy", "mixed", "detected")


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def load_cache(cache_dir, name, protocol=None, protocol_sha256=None):
    """Load one feature/detection cache and return id-indexed records."""
    root = Path(cache_dir)
    with np.load(root / f"features-{name}.npz", allow_pickle=False) as data:
        ids = data["sample_ids"].tolist()
        legacy = np.asarray(data["legacy_embeddings"], dtype=np.float64)
        detected = np.asarray(data["detected_embeddings"], dtype=np.float64)
        available = np.asarray(data["available"], dtype=bool)
    if legacy.ndim != 2 or legacy.shape[0] != len(ids) or detected.shape != legacy.shape or available.shape != (len(ids),) or not np.isfinite(legacy).all() or not np.isfinite(detected).all():
        raise ValueError("invalid feature cache shapes")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_ids in feature cache")
    statuses = {}
    detections = root / "detections.jsonl"
    if detections.exists():
        with detections.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row["sample_id"] in statuses:
                        raise ValueError("duplicate sample_id in detections.jsonl")
                    statuses[row["sample_id"]] = row
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    model_meta = metadata.get("models", {}).get(name, {})
    if model_meta.get("embedding_dim") is not None and legacy.shape[1] != int(model_meta["embedding_dim"]):
        raise ValueError("feature cache embedding dimension does not match model metadata")
    if protocol is not None:
        if metadata.get("partition") != "development":
            raise ValueError("feature cache partition must be development")
        expected = {r["sample_id"] for group in protocol["development"].values() for r in group}
        if set(ids) != expected:
            raise ValueError("feature cache sample_ids do not match protocol development")
        protocol_sha = metadata.get("protocol_sha256")
        if protocol_sha and protocol_sha256:
            if protocol_sha != protocol_sha256:
                raise ValueError("feature cache protocol sha256 does not match protocol")
    if detections.exists():
        if set(statuses) != set(ids):
            raise ValueError("detections.jsonl sample_ids do not match feature cache")
        for sample, row in statuses.items():
            if sample in ids and bool(available[ids.index(sample)]) != (row.get("status") == "ok"):
                raise ValueError(f"feature availability/status mismatch for {sample}")
    return {sample: {"legacy": legacy[i], "detected": detected[i], "available": bool(available[i]),
                     "status": statuses.get(sample, {}).get("status"),
                     "detection": statuses.get(sample, {})}
            for i, sample in enumerate(ids)}


def _embedding(record, mode, role):
    use_detected = mode == "detected" or (mode == "mixed" and role != "gallery")
    if use_detected and not record["available"]:
        return None
    return record["detected"] if use_detected else record["legacy"]


def _failure(record):
    status = record.get("detection", {}).get("status")
    return status if status in {"no_cat", "unusable"} else "unusable"


def _prediction(row, record, mode, role, gallery):
    vector = _embedding(record, mode, role)
    if vector is None:
        return {"sample_id": row["sample_id"], "split": role, "cat_id": row["cat_id"], "status": _failure(record), "score": None,
                "predicted_id": None, "available": False, "threshold": None, "accepted": False}
    result = rank_embedding(vector, gallery)
    return {"sample_id": row["sample_id"], "split": role, "cat_id": row["cat_id"], "status": "ok", "available": True,
            "predicted_id": result["predicted_id"], "score": result["score"],
            "similarity": result["similarity"], "runner_up_similarity": result["runner_up_similarity"], "threshold": None, "accepted": False}


def evaluate_episode(episode, records, mode="legacy", target_fpir=0.01):
    """Evaluate one episode, retaining failed planned queries in denominators."""
    if mode not in MODES:
        raise ValueError("unknown crop mode")
    split = episode["split"]
    all_ids = [r["sample_id"] for group in split.values() for r in group]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("episode sample_ids must be unique across roles")
    gallery_ids = {r["cat_id"] for r in split["gallery"]}
    known_ids = {r["cat_id"] for r in split["known"]}
    calibration_ids = {r["cat_id"] for r in split["calibration_unknown"]}
    unknown_ids = {r["cat_id"] for r in split["unknown"]}
    if known_ids - gallery_ids or gallery_ids & (calibration_ids | unknown_ids) or calibration_ids & unknown_ids:
        raise ValueError("episode identity groups violate gallery/known/unknown separation")
    gallery_rows = split["gallery"]
    gallery_vectors = []
    enrollment_available = {}
    for row in gallery_rows:
        enrollment_available[row["cat_id"]] = enrollment_available.get(row["cat_id"], 0)
        vector = _embedding(records[row["sample_id"]], mode, "gallery")
        gallery_vectors.append(vector)
        if vector is not None: enrollment_available[row["cat_id"]] += 1
    if any(v is None for v in gallery_vectors):
        return {"status": "enrollment_incomplete", "metrics": None, "predictions": [],
                "enrollment_planned": len(gallery_rows), "enrollment_available": enrollment_available}
    gallery = build_gallery(gallery_rows, {r["sample_id"]: v for r, v in zip(gallery_rows, gallery_vectors)},
                            aggregation="prototype", score_mode="cosine")
    groups = {}
    for role in ("calibration_unknown", "known", "unknown"):
        groups[role] = [_prediction(row, records[row["sample_id"]], mode, role, gallery) for row in split[role]]
    calibration = [r["score"] for r in groups["calibration_unknown"] if r["available"]]
    if not calibration:
        return {"status": "calibration_unavailable", "metrics": None, "predictions": groups}
    threshold = threshold_at_fpir(calibration, target_fpir)
    known = groups["known"]; unknown = groups["unknown"]
    known_ok = [r for r in known if r["available"]]
    unknown_ok = [r for r in unknown if r["available"]]
    accepted_known = [r for r in known_ok if r["score"] >= threshold]
    accepted_unknown = [r for r in unknown_ok if r["score"] >= threshold]
    for group in groups.values():
        for row in group:
            row["threshold"] = threshold
            row["accepted"] = bool(row["available"] and row["score"] >= threshold)
    total_k, total_u = len(known), len(unknown)
    metrics = {
        "threshold": threshold, "n_calibration_unknown": len(groups["calibration_unknown"]),
        "n_calibration_available": len(calibration), "n_known": total_k, "n_unknown": total_u,
        "n_known_available": len(known_ok), "n_unknown_available": len(unknown_ok),
        "dir": sum(r["predicted_id"] == r["cat_id"] for r in accepted_known) / total_k,
        "conditional_dir": (sum(r["predicted_id"] == r["cat_id"] for r in accepted_known) / len(known_ok)) if known_ok else None,
        "known_misidentification_rate": sum(r["predicted_id"] != r["cat_id"] for r in accepted_known) / total_k,
        "identity_reject": sum(r["score"] < threshold for r in known_ok) / total_k,
        "preprocessing_failure": (total_k - len(known_ok)) / total_k,
        "fpir": len(accepted_unknown) / total_u,
        "conditional_fpir": (len(accepted_unknown) / len(unknown_ok)) if unknown_ok else None,
        "unknown_preprocessing_failure": (total_u - len(unknown_ok)) / total_u,
    }
    return {"status": "ok", "metrics": metrics, "predictions": groups,
            "enrollment_planned": len(gallery_rows), "enrollment_available": enrollment_available}


def run(protocol, cache_dir, out, model_name, partition="development", target_fpir=0.01, protocol_sha256=None):
    if partition != "development":
        raise ValueError("crop study is restricted to the development partition")
    records = load_cache(cache_dir, model_name, protocol, protocol_sha256)
    suite = build_paired_suite(protocol, partition=partition, seeds=range(10), gallery_sizes=(2, 5), enrollments=(3, 5))
    destination = Path(out)
    all_results = {}
    for mode in MODES:
        results = []
        for episode in suite:
            evaluated = evaluate_episode(episode, records, mode, target_fpir)
            results.append({"episode_id": episode["episode_id"], "gallery_size": episode["gallery_size"],
                            "enrollment": episode["enrollment"], **evaluated})
        complete = [r for r in results if r["status"] == "ok"]
        def mean_field(field):
            values = [r["metrics"][field] for r in complete if r["metrics"].get(field) is not None]
            return {"mean": float(np.mean(values)) if values else None, "n_episodes": len(values)}
        fields = ("dir", "conditional_dir", "fpir", "conditional_fpir", "known_misidentification_rate",
                  "identity_reject", "preprocessing_failure", "unknown_preprocessing_failure")
        def summarize_rows(subset):
            return {field: {"mean": float(np.mean(values)) if (values := [r["metrics"][field] for r in subset if r["metrics"].get(field) is not None]) else None,
                             "n_episodes": len(values)} for field in fields}
        summary = {"mode": mode, "n_episodes": len(results), "n_complete": len(complete),
                   "status_counts": {status: sum(r["status"] == status for r in results)
                                     for status in ("ok", "enrollment_incomplete", "calibration_unavailable")},
                   "enrollment_available_per_episode": [r.get("enrollment_available", {}) for r in results],
                   "repeated_images_note": "Episodes reuse fixed image-level pools; not independent samples.",
                   "metrics": {field: mean_field(field) for field in fields},
                   "by_setting": {f"g{g}-e{e}": summarize_rows([r for r in complete if r["gallery_size"] == g and r["enrollment"] == e])
                                  for g in (2, 5) for e in (3, 5)}
                              if complete else None}
        _save(destination / mode / "summary.json", summary)
        with (destination / mode / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for row in results: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        all_results[mode] = summary
    fixed = {"episode_id": "fixed-development", "split": protocol[partition]}
    for mode in MODES:
        fixed_result = evaluate_episode(fixed, records, mode, target_fpir)
        fixed_dir = destination / "fixed"; fixed_dir.mkdir(parents=True, exist_ok=True)
        _save(fixed_dir / f"{mode}.json", fixed_result)
        if fixed_result["status"] == "ok":
            # Keep the gallery in the same portable form consumed by runtime.
            rows = protocol[partition]["gallery"]
            vectors = {r["sample_id"]: _embedding(records[r["sample_id"]], mode, "gallery") for r in rows}
            gallery = build_gallery(rows, vectors, "prototype", "cosine")
            cache_meta = json.loads((Path(cache_dir) / "metadata.json").read_text(encoding="utf-8"))
            model_meta = cache_meta.get("models", {}).get(model_name, {})
            gallery.update({"model_sha256": model_meta.get("model_sha256"), "input_size": model_meta.get("input_size"),
                            "embedding_dim": model_meta.get("embedding_dim"), "threshold": fixed_result["metrics"]["threshold"],
                            "diagnostic_only": True, "crop_pipeline": {"mode": mode,
                            "detector_model_sha256": cache_meta.get("detector", {}).get("model_sha256"),
                            "detector_metadata_sha256": cache_meta.get("detector", {}).get("metadata_sha256"),
                            "crop_mode": cache_meta.get("crop_mode", "head"),
                            "crop_policy": cache_meta.get("crop_policy", "cache detected crop; legacy uses reviewed face crop")}})
            _save(fixed_dir / f"{mode}.gallery.json", gallery)
    # Common-mode macro is written from episode files without retaining all rows in memory.
    mode_rows = {}
    for mode in MODES:
        path = destination / mode / "episodes.jsonl"
        mode_rows[mode] = {row["episode_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())}
    common = set.intersection(*(set(v) for v in mode_rows.values()))
    common_complete = {eid for eid in common if all(mode_rows[m][eid]["status"] == "ok" for m in MODES)}
    common_fields = ("dir", "fpir", "conditional_dir", "conditional_fpir", "known_misidentification_rate",
                     "identity_reject", "preprocessing_failure", "unknown_preprocessing_failure")
    def common_mean(mode, field):
        values = [mode_rows[mode][e]["metrics"].get(field) for e in common_complete
                  if mode_rows[mode][e]["metrics"].get(field) is not None]
        return {"mean": float(np.mean(values)) if values else None, "n_episodes": len(values)}
    _save(destination / "common-summary.json", {"n_common_episodes": len(common_complete), "episode_ids": sorted(common_complete),
          "modes": {m: {field: common_mean(m, field) for field in common_fields}
                    for m in MODES}})
    _save(destination / "summary.json", {"partition": partition, "target_fpir": target_fpir, "modes": all_results,
                                          "warning": "Repeated images are not independent samples."})
    return all_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True); parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--model-name", required=True)
    parser.add_argument("--partition", default="development"); parser.add_argument("--target-fpir", type=float, default=0.01)
    args = parser.parse_args(argv)
    return run(json.loads(args.protocol.read_text(encoding="utf-8")), args.cache_dir, args.out, args.model_name, args.partition, args.target_fpir,
               _digest(args.protocol))


if __name__ == "__main__": main()
