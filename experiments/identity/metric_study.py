"""Train-only linear feature adaptation; frozen full-frame development episodes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.detection.crop_study import evaluate_episode, load_cache
from .episodes import build_paired_suite
from .metric import apply_transform, fit_transform
from .runtime import Encoder, embed_rows, model_digest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/runs/edge160/protocol.json"
TRAIN_PROTOCOL = ROOT / "experiments/runs/head-adaptation/prepared/head-protocol.json"
CACHE = ROOT / "experiments/runs/crop-study/cache"
MODEL = ROOT / "experiments/runs/backbone-study/large224/model.onnx"
MODEL_NAME = "large224"
FIELDS = ("dir", "fpir", "conditional_dir", "conditional_fpir",
          "known_misidentification_rate", "identity_reject",
          "preprocessing_failure", "unknown_preprocessing_failure")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def training_rows(protocol, prepared):
    """Only existing training samples, with matching labels and detector crops."""
    rows = prepared["train"]
    original = {r["sample_id"]: r for r in protocol["train"]}
    held = {r["cat_id"] for key in ("development", "test")
            for group in protocol[key].values() for r in group}
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate training sample IDs")
    for row in rows:
        source = original.get(row["sample_id"])
        if source is None or source["cat_id"] != row["cat_id"] or row["cat_id"] in held:
            raise ValueError("transform fitting must use original training identities/samples only")
        if not row.get("crop_box"):
            raise ValueError("training features require the frozen detector head crop")
    if prepared["training_crop_pipeline"]["mode"] != "head":
        raise ValueError("this experiment is restricted to head crops")
    return rows


def adapted_records(records, transform):
    output = {}
    for sample, record in records.items():
        row = dict(record)
        # Missing detections are not features; retain the original failure.
        if record["available"]:
            row["detected"] = apply_transform(record["detected"], transform)
        output[sample] = row
    return output


def evaluate(records, suite, fixed):
    episodes = [{"episode_id": e["episode_id"], "gallery_size": e["gallery_size"],
                 "enrollment": e["enrollment"],
                 **evaluate_episode(e, records, "detected", 0.01)} for e in suite]
    complete = [e for e in episodes if e["status"] == "ok"]
    summary = {"n_complete": len(complete), "n_planned": len(episodes),
               "episode_ids": [e["episode_id"] for e in complete],
               "metrics": {key: float(np.mean([e["metrics"][key] for e in complete]))
                           for key in FIELDS}}
    return episodes, summary, evaluate_episode(fixed, records, "detected", 0.01)


def benchmark_transform(vectors, transform):
    """Batch-one transform cost only, without CNN, images, or gallery matching."""
    timings = []
    for index in range(1020):
        started = time.perf_counter()
        apply_transform(vectors[index % len(vectors)], transform)
        if index >= 20:
            timings.append((time.perf_counter() - started) * 1000)
    return {"runs": len(timings), "warmup": 20,
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)),
            "array_bytes": int(np.asarray(transform["mean"]).nbytes +
                               np.asarray(transform["projection"]).nbytes),
            "scope": "float64 transform only, cached features, batch one; not full pipeline or board timing"}


def run(out, threads=2):
    if threads < 1:
        raise ValueError("threads must be positive")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be empty or new")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    prepared = json.loads(TRAIN_PROTOCOL.read_text(encoding="utf-8"))
    rows = training_rows(protocol, prepared)
    metadata = json.loads((CACHE / "metadata.json").read_text(encoding="utf-8"))
    model_sha = model_digest(MODEL)
    if metadata["models"][MODEL_NAME]["model_sha256"] != model_sha:
        raise ValueError("development cache uses a different identity model")
    pipeline = prepared["training_crop_pipeline"]
    if (pipeline["detector_sha256"] != metadata["detector"]["model_sha256"] or
            pipeline["detector_metadata_sha256"] != metadata["detector"]["metadata_sha256"] or
            metadata.get("crop_mode", "head") != "head"):
        raise ValueError("training and development crops use different detector settings")
    records = load_cache(CACHE, MODEL_NAME, protocol, model_digest(PROTOCOL))
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoder = Encoder(MODEL, threads)
    features = embed_rows(encoder, rows, ROOT / "data")
    x = np.asarray([features[r["sample_id"]] for r in rows], dtype=np.float64)
    labels = [r["cat_id"] for r in rows]
    np.savez_compressed(out / "train-features.npz", embeddings=x,
                        sample_ids=[r["sample_id"] for r in rows], cat_ids=labels)
    extraction_sec = time.perf_counter() - started
    started = time.perf_counter()
    learned = fit_transform(x, labels, dimensions=128, shrinkage=0.1)
    fit_sec = time.perf_counter() - started
    np.savez_compressed(out / "transform.npz", mean=learned["mean"],
                        projection=learned["projection"])
    save(out / "transform-benchmark.json", benchmark_transform(x, learned))
    transforms = {"raw": None, "center": {"mean": learned["mean"], "projection": None},
                  "whiten128": learned}
    suite = build_paired_suite(protocol, "development", seeds=range(10),
                               gallery_sizes=(2, 5), enrollments=(3, 5))
    fixed = {"episode_id": "fixed-development", "split": protocol["development"]}
    reports, results = {}, {}
    for name, transform in transforms.items():
        adapted = records if transform is None else adapted_records(records, transform)
        episodes, summary, fixed_result = evaluate(adapted, suite, fixed)
        reports[name], results[name] = summary, episodes
        save(out / name / "summary.json", summary)
        save(out / name / "fixed.json", fixed_result)
        with (out / name / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for episode in episodes:
                stream.write(json.dumps(episode, ensure_ascii=False, allow_nan=False) + "\n")
        print(name, json.dumps(summary["metrics"]), flush=True)
    common = reports["raw"]["episode_ids"]
    if any(report["episode_ids"] != common for report in reports.values()):
        raise AssertionError("transformation changed available episodes")
    baseline = {e["episode_id"]: e for e in results["raw"] if e["status"] == "ok"}
    paired = {}
    for name in ("center", "whiten128"):
        paired[name] = {}
        for key in ("dir", "fpir"):
            deltas = [e["metrics"][key] - baseline[e["episode_id"]]["metrics"][key]
                      for e in results[name] if e["status"] == "ok"]
            paired[name][key] = {"mean_delta": float(np.mean(deltas)),
                                 "increased": sum(v > 0 for v in deltas),
                                 "decreased": sum(v < 0 for v in deltas),
                                 "equal": sum(v == 0 for v in deltas)}
    report = {"fit_partition": "train only; validation/development/test not used for fitting",
              "train_images": len(rows), "train_identities": len(set(labels)),
              "train_sample_ids": [r["sample_id"] for r in rows],
              "model_sha256": model_sha, "training_protocol_sha256": model_digest(TRAIN_PROTOCOL),
              "crop_pipeline": pipeline, "dimensions": 128, "shrinkage": 0.1,
              "fit_sec": fit_sec, "extraction_sec": extraction_sec, "threads": threads,
              "models": reports, "paired": paired,
              "scope": "Exploratory image-level development; repeated episodes are not independent cats or events. No test image read."}
    save(out / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.out, args.threads)


if __name__ == "__main__":
    main()
