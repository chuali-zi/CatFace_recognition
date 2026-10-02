"""Frozen head-crop feature study on the fresh-identity protocol."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.detection.crop_study import evaluate_episode
from experiments.identity.metric import apply_transform
from experiments.identity.runtime import Encoder, model_digest, preprocess, read_image

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"large224": ROOT / "experiments/runs/backbone-study/large224/model.onnx",
          "small160": ROOT / "experiments/runs/edge160/baseline/model.onnx"}
TRANSFORM = ROOT / "experiments/runs/metric-study/large224/transform.npz"
FIELDS = ("dir", "conditional_dir", "fpir", "conditional_fpir", "known_misidentification_rate",
          "identity_reject", "preprocessing_failure", "unknown_preprocessing_failure")


def _load_rows(data_dir):
    data_dir = Path(data_dir)
    reviewed = {}
    with (data_dir / "reviewed.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["sample_id"] in reviewed:
                    raise ValueError("duplicate reviewed sample_id")
                reviewed[row["sample_id"]] = row
    if not reviewed:
        raise ValueError("reviewed manifest is empty")
    protocol = json.loads((data_dir / "protocol.json").read_text(encoding="utf-8"))
    episodes = protocol.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("protocol must contain a non-empty episodes list")
    include_ids = {sample for sample, row in reviewed.items() if row.get("evaluation_include") is True}
    for episode in episodes:
        episode_ids = [row["sample_id"] for group in episode["split"].values() for row in group]
        if set(episode_ids) != include_ids or len(episode_ids) != len(include_ids):
            raise ValueError("each episode must contain exactly the evaluation_include sample set")
        for group in episode["split"].values():
            for row in group:
                source = reviewed[row["sample_id"]]
                if row.get("cat_id") != source.get("cat_id") or row.get("path") != source.get("path"):
                    raise ValueError(f"protocol/reviewed row mismatch: {row['sample_id']}")
    return reviewed, episodes


def _path(data_dir, value):
    path = Path(value)
    candidates = [path if path.is_absolute() else ROOT / "data" / path,
                  Path(data_dir) / path]
    for candidate in candidates:
        if candidate.exists(): return candidate
    raise FileNotFoundError(value)


def _records(rows, data_dir, model, threads):
    encoder = Encoder(model, threads=threads)
    output = {}
    started = time.perf_counter()
    for row in rows:
        sample = row["sample_id"]
        status = row.get("status", "ok")
        include = row["evaluation_include"] is True
        record = {"available": False, "detected": np.zeros(encoder.metadata["embedding_dim"], dtype=np.float64),
                  "legacy": np.zeros(encoder.metadata["embedding_dim"], dtype=np.float64),
                  "detection": {"status": status if status in {"no_cat", "unusable"} else "unusable"}}
        if include and status == "ok":
            if not row.get("crop_box"):
                raise ValueError(f"ok reviewed row has no crop_box: {sample}")
            image = read_image(_path(data_dir, row.get("path") or row.get("original_path")), row.get("crop_box"))
            record["detected"] = encoder.encode_array(preprocess(image, encoder.size)).astype(np.float64)
            record["available"] = True; record["detection"] = {"status": "ok"}
        output[sample] = record
    return encoder, output, time.perf_counter() - started


def _adapt(records, transform):
    output = {}
    for sample, record in records.items():
        value = dict(record)
        if record["available"]: value["detected"] = apply_transform(record["detected"], transform)
        output[sample] = value
    return output


def _summary(results):
    complete = [row for row in results if row["status"] == "ok"]
    metrics = {}
    for field in FIELDS:
        values = [row["metrics"][field] for row in complete if row["metrics"].get(field) is not None]
        metrics[field] = {"mean": float(np.mean(values)) if values else None, "n_episodes": len(values)}
    return {"n_planned": len(results), "n_complete": len(complete), "metrics": metrics,
            "status_counts": {status: sum(row["status"] == status for row in results)
                              for status in ("ok", "enrollment_incomplete", "calibration_unavailable")}}


def _write_arm(out, name, records, episodes, fixed):
    results = []
    for episode in episodes:
        value = evaluate_episode(episode, records, "detected", target_fpir=0.01)
        results.append({"episode_id": episode["episode_id"], "gallery_size": episode.get("gallery_size"),
                        "enrollment": episode.get("enrollment"), **value})
    arm = Path(out) / name; arm.mkdir(parents=True, exist_ok=True)
    with (arm / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in results: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = _summary(results); summary["target_fpir"] = 0.01
    summary["fixed"] = fixed
    (arm / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def run(data_dir, out, threads=2):
    if threads < 1: raise ValueError("threads must be positive")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty")
    reviewed, episodes = _load_rows(data_dir)
    study_summary = json.loads((ROOT / "experiments/runs/metric-study/large224/summary.json").read_text(encoding="utf-8"))
    large_model_sha = model_digest(MODELS["large224"])
    if study_summary.get("model_sha256") != large_model_sha:
        raise ValueError("Large224 model does not match metric-study fit report")
    provenance = study_summary.get("crop_pipeline", {})
    fresh_meta = json.loads((Path(data_dir) / "metadata.json").read_text(encoding="utf-8"))
    if fresh_meta.get("crop_mode") != "head" or provenance.get("mode") != "head":
        raise ValueError("fresh crops and fitted transform must use head mode")
    for key in ("detector_sha256", "detector_metadata_sha256"):
        if fresh_meta.get(key) != provenance.get(key):
            raise ValueError(f"fresh metadata does not match frozen metric-study {key}")
    rows = list(reviewed.values())
    by_id = {row["sample_id"]: row for row in rows}
    all_episode_rows = [row for episode in episodes for group in episode["split"].values() for row in group]
    if any(row["sample_id"] not in by_id for row in all_episode_rows): raise ValueError("protocol row absent from reviewed.jsonl")
    model_results = {}; out.mkdir(parents=True, exist_ok=True)
    for name in ("large224", "small160"):
        model = MODELS[name]
        encoder, records, elapsed = _records(rows, data_dir, model, threads)
        model_dir = out / name; model_dir.mkdir(parents=True, exist_ok=True)
        available_ids = [sample for sample, record in records.items() if record["available"]]
        np.savez_compressed(model_dir / "features.npz", sample_ids=np.asarray(available_ids),
                            detected_embeddings=np.stack([records[sample]["detected"] for sample in available_ids]),
                            available=np.ones(len(available_ids), dtype=bool))
        fixed = {"episode_id": "protocol-fixed", "split": episodes[0]["split"]}
        if name == "large224":
            with np.load(TRANSFORM, allow_pickle=False) as data:
                transform = {"mean": data["mean"], "projection": data["projection"]}
            arms = {"large224_raw": records, "large224_center": _adapt(records, {"mean": transform["mean"], "projection": None}),
                    "large224_whiten128": _adapt(records, transform)}
        else:
            arms = {"small160_raw": records}
        model_results[name] = {}
        for arm, values in arms.items():
            model_results[name][arm] = _write_arm(model_dir, arm, values, episodes, evaluate_episode(fixed, values, "detected", 0.01))
            print(arm, json.dumps(model_results[name][arm]["metrics"]), flush=True)
        (model_dir / "metadata.json").write_text(json.dumps({"model": str(model), "model_sha256": model_digest(model),
            "input_size": encoder.size, "embedding_dim": encoder.metadata["embedding_dim"], "extracted_sec": elapsed,
            "n_reviewed": len(rows), "n_available": len(available_ids), "threads": threads,
            "transform_source": str(TRANSFORM) if name == "large224" else None,
            "transform_sha256": model_digest(TRANSFORM) if name == "large224" else None}, indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"models": model_results, "episodes": len(episodes), "target_fpir": 0.01,
        "scope": "fresh publisher-label image-level confirmation; old identity test not used for model selection; frozen head crops and no detector rerun"}, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return model_results


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(); run(args.data_dir, args.out, args.threads)


if __name__ == "__main__": main()
