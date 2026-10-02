"""Diagnostic-only Haar cat-face replay and positive-frame coverage check."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .deploy import read_gallery
from .runtime import Encoder, match_gallery, preprocess


def resize_for_detection(image: Image.Image, max_side: int) -> tuple[Image.Image, tuple[float, float]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max_side < 1:
        raise ValueError("max-side must be positive")
    scale = min(1.0, max_side / max(image.size))
    if scale == 1.0:
        return image, (1.0, 1.0)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR), (size[0] / image.width, size[1] / image.height)


def detect_candidates(image: Image.Image, max_side: int = 640, threads: int = 2) -> tuple[list[list[int]], tuple[float, float]]:
    import cv2

    if threads < 1:
        raise ValueError("threads must be positive")
    cv2.setNumThreads(threads)
    resized, (sx, sy) = resize_for_detection(image, max_side)
    gray = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalcatface_extended.xml")
    )
    if cascade.empty():
        raise RuntimeError("OpenCV cat-face cascade could not be loaded")
    found = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE
    )
    boxes = []
    for x, y, width, height in found:
        boxes.append([round(x / sx), round(y / sy),
                      round((x + width) / sx), round((y + height) / sy)])
    return boxes, (sx, sy)


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def replay(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    encoder = Encoder(args.model, args.threads)
    gallery = read_gallery(args.gallery, encoder)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        boxes, scales = detect_candidates(source, args.max_side, args.threads)
        result = {"status": "no_cat", "diagnostic_only": True}
        if len(boxes) > 1:
            result = {"status": "unusable", "reason": "multiple cats/candidates", "diagnostic_only": True}
        elif len(boxes) == 1:
            box = boxes[0]
            crop = image.crop(tuple(box))
            embedding = encoder.encode_array(preprocess(crop, encoder.size))
            candidate_id, score = match_gallery(embedding, gallery)
            accepted = score >= gallery["threshold"]
            result = {"status": "known" if accepted else "unknown",
                      "cat_id": candidate_id if accepted else None,
                      "top_candidate_id": candidate_id, "score": score, "diagnostic_only": True}
        result.update({"image": str(args.image), "detector": "opencv-haarcatface-extended",
                       "detector_candidates": boxes,
                       "resize_scales": scales,
                       "scope": "Detected crop; end-to-end crop/identity threshold is not calibrated",
                       "elapsed_ms": (time.perf_counter() - started) * 1000})
        return result


def _protocol_rows(protocol: dict, partition: str) -> list[dict]:
    split = protocol.get(partition)
    if not isinstance(split, dict):
        raise ValueError(f"protocol has no partition: {partition}")
    return [row for group in ("gallery", "known", "calibration_unknown", "unknown")
            for row in split.get(group, [])]


def check(args: argparse.Namespace) -> dict:
    manifest_rows = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("original_path") and row.get("face_box"):
                    manifest_rows[row["path"]] = row
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = [row for row in _protocol_rows(protocol, args.partition) if row.get("path") in manifest_rows]
    if args.limit:
        rows = rows[:args.limit]
    output_rows, timings, matched = [], [], 0
    for row in rows:
        record = manifest_rows[row["path"]]
        original = args.manifest.parent / record["original_path"]
        started = time.perf_counter()
        with Image.open(original) as image:
            boxes, _ = detect_candidates(image, args.max_side, args.threads)
            best = max((iou(box, record["face_box"]) for box in boxes), default=0.0)
            timings.append((time.perf_counter() - started) * 1000)
        matched += best >= 0.5
        output_rows.append({"sample_id": row.get("sample_id"), "original_path": record["original_path"],
                            "reference_box": record["face_box"], "candidates": boxes,
                            "best_iou": best, "matched_iou_0.5": best >= 0.5})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output_rows), encoding="utf-8")
    summary = {"diagnostic_only": True, "biased_positive_replay": True,
               "partition": args.partition, "n": len(rows), "matched_iou_0.5": matched,
               "match_rate_iou_0.5": matched / len(rows) if rows else 0.0,
               "multi_candidate": sum(len(r["candidates"]) > 1 for r in output_rows),
               "missed": sum(not r["candidates"] for r in output_rows),
               "p50_ms": statistics.median(timings) if timings else None,
               "p95_ms": sorted(timings)[max(0, round(len(timings) * .95) - 1)] if timings else None,
               "timing_scope": "image decode + EXIF/RGB conversion + resize + Haar detection; cascade is loaded per image",
               "threads": args.threads, "max_side": args.max_side,
               "detector_parameters": {"scaleFactor": 1.1, "minNeighbors": 3, "minSize": [32, 32]},
               "rows_jsonl": str(jsonl),
               "limitations": "Positive frames were selected through the same Haar candidate workflow; face_box is a crop annotation, not independent detection ground truth. Not recall/AP or a real-visit estimate."}
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--image", type=Path, required=True)
    replay_parser.add_argument("--model", type=Path, required=True)
    replay_parser.add_argument("--gallery", type=Path, required=True)
    replay_parser.add_argument("--threads", type=int, default=2)
    replay_parser.add_argument("--max-side", type=int, default=640)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--manifest", type=Path, required=True)
    check_parser.add_argument("--protocol", type=Path, required=True)
    check_parser.add_argument("--partition", choices=("development", "test"), default="development")
    check_parser.add_argument("--out", type=Path, required=True)
    check_parser.add_argument("--max-side", type=int, default=640)
    check_parser.add_argument("--threads", type=int, default=2)
    check_parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    if getattr(args, "threads", 1) < 1 or args.max_side < 1 or getattr(args, "limit", 0) < 0:
        parser.error("threads/max-side must be positive and limit must be non-negative")
    result = replay(args) if args.command == "replay" else check(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
