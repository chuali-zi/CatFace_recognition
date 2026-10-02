"""Cache old/detected-crop identity features on fixed development originals."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .evaluate import iou
from .onnx import Detector, single_crop, crop_policy
from experiments.identity.runtime import Encoder, model_digest, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name=identity-model.onnx")
    parser.add_argument("--manifest", type=Path, default=Path("data/clean.identity.jsonl"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--crop-mode", choices=["head", "face_core"], default="head")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output must be empty; use a new directory for changed inputs")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = sorted([row for group in protocol["development"].values() for row in group], key=lambda r: r["sample_id"])
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("development contains duplicate sample IDs")
    manifest = {row["path"]: row for row in (
        json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip())}
    detector = Detector(args.detector, args.threads)
    encoders, metadata_models = {}, {}
    for spec in args.models:
        name, separator, path = spec.partition("=")
        if not separator or not name.isidentifier() or name in encoders:
            parser.error("models need unique identifier names: name=path")
        encoder = Encoder(Path(path), args.threads)
        if {row["cat_id"] for row in rows} & set(encoder.metadata.get("training_identities", [])):
            raise ValueError("identity model training labels overlap development identities")
        encoders[name] = encoder
        metadata_models[name] = {"path": path, "model_sha256": model_digest(Path(path)),
            "input_size": encoder.size, "embedding_dim": encoder.metadata["embedding_dim"],
            "architecture": encoder.metadata.get("architecture"),
            "model_metadata_sha256": model_digest(Path(path).with_suffix(".json"))}
    args.out.mkdir(parents=True, exist_ok=True)
    old_vectors = {name: [] for name in encoders}
    new_vectors = {name: [] for name in encoders}
    records, source_state, available = [], [], []
    started = time.perf_counter()
    with (args.out / "detections.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            record = manifest[row["path"]]
            if record["cat_id"] != row["cat_id"]:
                raise ValueError("protocol and identity manifest labels disagree")
            original_path = args.manifest.parent / record["original_path"]
            legacy_path = args.manifest.parent / record["path"]
            original_stat, legacy_stat = original_path.stat(), legacy_path.stat()
            source_state.append([row["sample_id"], record["original_path"], original_stat.st_size,
                                 original_stat.st_mtime_ns, legacy_stat.st_size, legacy_stat.st_mtime_ns])
            with Image.open(original_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            boxes, scores = detector.detect(image)
            status, crop_box, crop = single_crop(image, boxes, args.crop_mode)
            available.append(status == "ok")
            with Image.open(legacy_path) as opened:
                legacy = opened.convert("RGB")
            for name, encoder in encoders.items():
                old_vectors[name].append(encoder.encode_array(preprocess(legacy, encoder.size)))
                new_vectors[name].append(encoder.encode_array(preprocess(crop, encoder.size)) if crop is not None
                                         else np.zeros(int(encoder.metadata["embedding_dim"]), dtype=np.float32))
            observation = {"sample_id": row["sample_id"], "cat_id": row["cat_id"],
                "original_path": record["original_path"], "legacy_path": record["path"],
                "width": image.width, "height": image.height, "status": status,
                "boxes": boxes.tolist(), "scores": scores.tolist(), "crop_box": crop_box,
                "reference_face_box": record["face_box"],
                "reference_iou_diagnostic_only": iou(crop_box, record["face_box"]) if crop_box else None}
            records.append(observation)
            stream.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(rows):
                print(f"cached {index}/{len(rows)} originals and {len(encoders)} identity models", flush=True)
    for name in encoders:
        np.savez_compressed(args.out / f"features-{name}.npz",
            sample_ids=np.asarray([row["sample_id"] for row in rows]), available=np.asarray(available, dtype=bool),
            legacy_embeddings=np.stack(old_vectors[name]), detected_embeddings=np.stack(new_vectors[name]))
    metadata = {"partition": "development", "n_images": len(rows),
        "models": metadata_models, "source_state": source_state,
        "protocol_sha256": model_digest(args.protocol), "manifest_sha256": model_digest(args.manifest),
        "detector": {"path": str(args.detector), "model_sha256": model_digest(args.detector),
                     "metadata_sha256": model_digest(args.detector.with_suffix(".onnx.json")),
                     "metadata": detector.metadata},
        "crop_mode": args.crop_mode, "crop_policy": crop_policy(args.crop_mode),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "threads": args.threads, "elapsed_sec": time.perf_counter() - started,
        "limitations": "Old identity pool was Haar-selected; diagnostic crop shift, not independent detection recall or event-level identity validation. Unavailable rows have zero cache padding, never valid features."}
    (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_images": len(rows), "status_counts": metadata["status_counts"],
                      "elapsed_sec": metadata["elapsed_sec"]}, indent=2))


if __name__ == "__main__":
    main()
