"""Prepare matched legacy/head-crop identity training protocols offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from .onnx import Detector, crop_policy, single_crop


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare_rows(manifest_rows, protocol, data_root: Path, detector):
    """Return records and matched train/validation rows without writing images."""
    if len({r.get("path") for r in manifest_rows}) != len(manifest_rows):
        raise ValueError("manifest paths must be unique")
    by_path = {r["path"]: r for r in manifest_rows}
    selected = [(split, row) for split in ("train", "validation") for row in protocol.get(split, [])]
    if len({r["sample_id"] for _, r in selected}) != len(selected) or len({r["path"] for _, r in selected}) != len(selected):
        raise ValueError("train and validation sample_ids/paths must be unique across roles")
    held_ids = {r["cat_id"] for key in ("development", "test") for rows in protocol.get(key, {}).values() for r in rows}
    if {r["cat_id"] for _, r in selected} & held_ids:
        raise ValueError("training identities overlap development/test identities")
    records, successes = [], {"train": [], "validation": []}
    for split, row in selected:
        source = by_path.get(row["path"])
        if source is None:
            raise ValueError(f"protocol path missing from manifest: {row['path']}")
        if source.get("cat_id") != row.get("cat_id"):
            raise ValueError(f"manifest and protocol identity labels disagree for {row['path']}")
        original = data_root / source["original_path"]
        with Image.open(original) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            boxes, scores = detector.detect(image)
            status, crop_box, _ = single_crop(image, boxes, "head")
        if status == "ok" and source.get("face_box"):
            box = source["face_box"]
            cx, cy = (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2
            if not (crop_box[0] <= cx <= crop_box[2] and crop_box[1] <= cy <= crop_box[3]):
                status = "label_target_mismatch"
        record = {"original_path": source["original_path"], "legacy_path": source["path"], "sample_id": row["sample_id"], "cat_id": row["cat_id"], "split": split, "width": width, "height": height, "boxes": boxes.tolist(), "scores": scores.tolist(), "crop_box": crop_box, "status": status}
        try:
            stat = original.stat(); record.update({"original_size": stat.st_size, "original_mtime_ns": stat.st_mtime_ns})
        except FileNotFoundError:
            raise ValueError(f"missing original image: {original}")
        records.append(record)
        if status == "ok":
            successes[split].append((row, source, crop_box))
        if len(records) % 100 == 0:
            print(f"prepared {len(records)} train/validation originals", flush=True)
    train_ids = {r[0]["cat_id"] for r in successes["train"]}
    val_ids = {r[0]["cat_id"] for r in successes["validation"]}
    kept_ids = train_ids & val_ids
    kept = {split: [item for item in successes[split] if item[0]["cat_id"] in kept_ids] for split in successes}
    head, legacy = [], []
    for split in ("train", "validation"):
        for row, source, box in kept[split]:
            head.append({**row, "path": source["original_path"], "crop_box": box, "split": split})
            legacy.append({**row, "path": source["path"], "split": split})
    if [(r["sample_id"], r["cat_id"]) for r in head] != [(r["sample_id"], r["cat_id"]) for r in legacy]:
        raise AssertionError("head and legacy protocols diverged")
    return records, head, legacy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1: parser.error("threads must be positive")
    if args.out.exists() and any(args.out.iterdir()): parser.error("out must be empty or not exist")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    detector = Detector(args.detector, args.threads)
    records, head_rows, legacy_rows = prepare_rows(manifest_rows, protocol, args.manifest.parent, detector)
    base = dict(protocol); base["train"], base["validation"] = [], []
    # Preserve development/test verbatim for identity-isolation auditing only.
    head_protocol = dict(base); head_protocol["train"] = [r for r in head_rows if r.get("split") == "train"]; head_protocol["validation"] = [r for r in head_rows if r.get("split") == "validation"]
    legacy_protocol = dict(base); legacy_protocol["train"] = [r for r in legacy_rows if r.get("split") == "train"]; legacy_protocol["validation"] = [r for r in legacy_rows if r.get("split") == "validation"]
    for prepared in (head_protocol, legacy_protocol):
        source_counts = prepared.get("counts")
        prepared["source_counts"] = source_counts
        prepared["training_counts"] = {"train": len(prepared["train"]), "validation": len(prepared["validation"]), "total": len(prepared["train"]) + len(prepared["validation"])}
        prepared["counts"] = {**(source_counts or {}), "train": len(prepared["train"]), "validation": len(prepared["validation"])}
        total = len(prepared["train"]) + len(prepared["validation"]) + sum(len(rows) for key in ("development", "test") for rows in prepared.get(key, {}).values())
        prepared["counts"].update(total=total, assigned=total)
    train_source_ids = {r["cat_id"] for r in protocol.get("train", [])}; val_source_ids = {r["cat_id"] for r in protocol.get("validation", [])}
    devtest_ids = {r.get("cat_id") for key in ("development", "test") for value in protocol.get(key, {}).values() for r in value}
    if train_source_ids & devtest_ids or val_source_ids & devtest_ids:
        raise ValueError("training identities overlap development/test identities")
    status_counts = dict(Counter(r["status"] for r in records))
    provenance = {"mode": "head", "detector_sha256": detector.model_sha256, "detector_metadata_sha256": detector.metadata_sha256, "crop_policy": crop_policy("head"), "status_counts": status_counts, "kept_identities": sorted({r["cat_id"] for r in head_rows}), "excluded_identities": sorted((train_source_ids | val_source_ids) - set(r["cat_id"] for r in head_rows))}
    head_protocol["training_crop_pipeline"] = provenance
    legacy_protocol["training_crop_pipeline"] = {**provenance, "mode": "legacy", "crop_policy": "Reviewed legacy PNG crops; same detector-selected training sample IDs as head arm"}
    (args.out / "head-protocol.json").write_text(json.dumps(head_protocol, indent=2), encoding="utf-8")
    (args.out / "legacy-protocol.json").write_text(json.dumps(legacy_protocol, indent=2), encoding="utf-8")
    (args.out / "detections.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    metadata = {"n_input": len(records), "n_kept": len(head_rows), "n_kept_identities": len(provenance["kept_identities"]), "status_counts": status_counts, "detector_sha256": detector.model_sha256, "detector_metadata_sha256": detector.metadata_sha256, "crop_policy": crop_policy("head"), "protocol_sha256": _sha256(args.protocol), "manifest_sha256": _sha256(args.manifest), "training_counts": {"train_original": sum(r["split"] == "train" for r in records), "validation_original": sum(r["split"] == "validation" for r in records), "train_kept": len(head_protocol["train"]), "validation_kept": len(head_protocol["validation"])}, "excluded_identities": provenance["excluded_identities"]}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
