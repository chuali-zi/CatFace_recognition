"""Fixed containment-suppression diagnostic on old train/development images.

This does not change detector metadata, enrollment crops, or the runtime default.
No identity test or fresh-identity images are used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from experiments.identity.runtime import read_image
from .containment import suppress_contained
from .evaluate import evaluate_rows
from .onnx import Detector

ROOT = Path(__file__).resolve().parents[2]
DETECTION = ROOT / "experiments/runs/detection-cpu"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare(row, dataset, threshold):
    boxes = np.asarray(row["boxes"], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(row["scores"], dtype=np.float64)
    keep = suppress_contained(boxes, scores, threshold)
    return {"sample_id": row["sample_id"], "dataset": dataset,
            "path": row.get("original_path", row.get("path")),
            "boxes": boxes.tolist(), "scores": scores.tolist(), "kept_indices": keep.tolist(),
            "candidate_boxes": boxes[keep].tolist(), "candidate_scores": scores[keep].tolist(),
            "changed": len(keep) != len(boxes), "ground_truth_boxes": row.get("ground_truth_boxes", [])}


def review_image(row, destination):
    original = read_image(ROOT / "data" / row["path"])
    sheet = Image.new("RGB", (1000, 440), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 4), f"{row['dataset']} | {Path(row['path']).name} | changed={row['changed']}", fill="black")
    for panel, boxes in enumerate((row["boxes"], row["candidate_boxes"])):
        thumb = ImageOps.contain(original, (490, 390))
        painter = ImageDraw.Draw(thumb)
        scale_x, scale_y = thumb.width / original.width, thumb.height / original.height
        for box in row["ground_truth_boxes"]:
            painter.rectangle([box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y], outline="yellow", width=2)
        for index, box in enumerate(boxes):
            xy = [box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y]
            painter.rectangle(xy, outline="red" if panel == 0 else "lime", width=2)
            painter.text((xy[0] + 2, xy[1] + 2), str(index), fill="white", stroke_width=1, stroke_fill="black")
        x = panel*500 + 5
        sheet.paste(thumb, (x, 32))
        scores = row["scores"] if panel == 0 else row["candidate_scores"]
        draw.text((x, 420), f"{'before' if panel == 0 else 'after'} scores=" + ",".join(f"{s:.3f}" for s in scores), fill="black")
    sheet.save(destination, quality=94)


def summarize(records):
    def statuses(field):
        return dict(Counter("no_cat" if not r[field] else "ok" if len(r[field]) == 1 else "unusable" for r in records))
    return {"n_images": len(records), "before": statuses("boxes"), "after": statuses("candidate_boxes"),
            "changed_images": sum(r["changed"] for r in records),
            "removed_boxes": sum(len(r["boxes"]) - len(r["candidate_boxes"]) for r in records)}


def run(out, threads=2):
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir()
    threshold = 0.9
    save(out / "plan.json", {"containment_threshold": threshold, "selection": "one fixed candidate; no parameter search",
                            "sources": ["CatFLW train/development", "old identity train/validation/development"],
                            "excluded": ["CatFLW test", "identity test", "fresh-identities"],
                            "rule": "after existing score0.7/NMS0.55; greedy higher-score winner when intersection/min(area)>=0.9"})
    detector = Detector(DETECTION / "onnx/model.onnx", threads)
    if detector.metadata["score_threshold"] != 0.7 or detector.metadata["nms_threshold"] != 0.55:
        raise ValueError("study expects frozen score0.7/NMS0.55")
    manifest = read_rows(DETECTION / "data-v2/manifest.jsonl")
    review = [r for r in read_rows(DETECTION / "data-v2/reviewed.jsonl") if r["split"] == "development"]
    datasets = {}
    for split in ("train", "development"):
        rows = [r for r in manifest if r["split"] == split]
        dataset = f"catflw_{split}"
        records = []
        if split == "development":
            path = DETECTION / "ssdlite-selected-development.jsonl"
            sidecar = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
            if sidecar["model_sha256"] != detector.model_sha256 or sidecar["metadata"] != detector.metadata:
                raise ValueError("development predictions use a different detector configuration")
            cached = {r["sample_id"]: r for r in read_rows(path)}
            if set(cached) != {r["sample_id"] for r in rows}:
                raise ValueError("development predictions do not match manifest")
        with (out / f"{dataset}.jsonl").open("w", encoding="utf-8") as stream:
            for index, row in enumerate(rows, 1):
                if split == "train":
                    boxes, scores = detector.detect(read_image(ROOT / "data" / row["path"]))
                    prediction = {"boxes": boxes.tolist(), "scores": scores.tolist()}
                else:
                    prediction = cached[row["sample_id"]]
                record = compare({**row, **prediction, "ground_truth_boxes": row["boxes"]}, dataset, threshold)
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if index % 200 == 0 or index == len(rows):
                    stream.flush()
                    print(f"{dataset}: {index}/{len(rows)}", flush=True)
        datasets[dataset] = records
    protocol = json.loads((ROOT / "experiments/runs/edge160/protocol.json").read_text(encoding="utf-8"))
    cache_paths = {"identity_train_validation": ROOT / "experiments/runs/head-adaptation/prepared/detections.jsonl",
                   "identity_development": ROOT / "experiments/runs/crop-study/cache/detections.jsonl"}
    for dataset, path in cache_paths.items():
        source = read_rows(path)
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        cached_model = metadata.get("detector_sha256", metadata.get("detector", {}).get("model_sha256"))
        cached_config = metadata.get("detector_metadata_sha256", metadata.get("detector", {}).get("metadata_sha256"))
        if cached_model != detector.model_sha256 or cached_config != detector.metadata_sha256:
            raise ValueError("identity crop cache uses a different detector configuration")
        expected = ({r["sample_id"] for role in ("train", "validation") for r in protocol[role]}
                    if dataset == "identity_train_validation" else
                    {r["sample_id"] for values in protocol["development"].values() for r in values})
        if {r["sample_id"] for r in source} != expected:
            raise ValueError("identity cache must retain ALL planned original samples, including failures")
        datasets[dataset] = [compare(r, dataset, threshold) for r in source]
        (out / f"{dataset}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in datasets[dataset]), encoding="utf-8")
    summary = {name: summarize(records) for name, records in datasets.items()}
    for split in ("train", "development"):
        dataset = f"catflw_{split}"
        rows = [r for r in manifest if r["split"] == split]
        sizes = {r["sample_id"]: (r["width"], r["height"]) for r in rows}
        summary[dataset]["annotation_metrics"] = {}
        for arm, field in (("before", "boxes"), ("after", "candidate_boxes")):
            predictions = {r["sample_id"]: r[field] for r in datasets[dataset]}
            metrics = evaluate_rows(rows, predictions, sizes)
            metrics.pop("per_image")
            summary[dataset]["annotation_metrics"][arm] = metrics
            if split == "development":
                metrics = evaluate_rows(review, predictions, sizes)
                metrics.pop("per_image")
                summary[dataset].setdefault("reviewed64_metrics", {})[arm] = metrics
    cases = []
    for records in datasets.values():
        for row in records:
            if len(row["boxes"]) > 1:
                path = out / "review" / f"{len(cases):03d}.jpg"
                review_image(row, path)
                cases.append({**row, "review_image": str(path), "visual_review": "pending"})
    (out / "cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases), encoding="utf-8")
    save(out / "summary.json", {"datasets": summary, "review_cases": len(cases), "threads": threads,
                                "detector_model_sha256": detector.model_sha256,
                                "detector_metadata_sha256": detector.metadata_sha256,
                                "scope": "Fixed geometric postprocess diagnostic, not identity accuracy or deployment validation"})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.out, args.threads)


if __name__ == "__main__":
    main()
