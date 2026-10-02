"""Deterministic, image-level diagnostic protocol for identity experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def _row(record: dict, manifest_dir: Path) -> dict:
    for key in ("path", "cat_id", "pixel_hash", "original_pixel_hash", "session_id"):
        if key not in record:
            raise ValueError(f"manifest row missing {key}")
    path = str(record["path"])
    cat = str(record["cat_id"])
    if not path or not cat:
        raise ValueError("manifest path and cat_id must be non-empty")
    # Keep paths relative to the manifest, as the input contract specifies.
    relative = Path(path).as_posix()
    return {"sample_id": relative, "path": relative, "cat_id": cat}


def _read_manifest(manifest: Path) -> list[dict]:
    rows = []
    pixel_seen: dict[str, str] = {}
    original_seen: dict[str, str] = {}
    with manifest.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at manifest line {number}") from exc
            if not isinstance(source, dict):
                raise ValueError(f"manifest line {number} is not an object")
            clean = _row(source, manifest.parent)
            for field, seen in (("pixel_hash", pixel_seen), ("original_pixel_hash", original_seen)):
                value = source[field]
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{field} must be a non-empty string")
                if value in seen:
                    raise ValueError(f"duplicate {field}: {value}")
                seen[value] = clean["sample_id"]
            rows.append(clean)
    if not rows:
        raise ValueError("manifest contains no rows")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("path/sample_id must be unique")
    return rows


def _split_ids(ids: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    if len(ids) < 3:
        raise ValueError("need at least three identities for train/development/test isolation")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * 0.60))
    n_dev = max(1, round(len(shuffled) * 0.20))
    if n_train + n_dev >= len(shuffled):
        n_train, n_dev = len(shuffled) - 2, 1
    return shuffled[:n_train], shuffled[n_train:n_train + n_dev], shuffled[n_train + n_dev:]


def _protocol_row(row: dict) -> dict:
    return {"sample_id": row["sample_id"], "path": row["path"], "cat_id": row["cat_id"]}


def build_protocol(manifest: Path, seed: int = 20260922, gallery_size: int = 5,
                   enrollment: int = 3) -> dict:
    manifest = Path(manifest)
    if gallery_size < 1 or enrollment < 1:
        raise ValueError("gallery_size and enrollment must be positive")
    rows = _read_manifest(manifest)
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[row["cat_id"]].append(row)
    train_ids, dev_pool, test_pool = _split_ids(list(by_id), seed)
    rng = random.Random(seed + 1)

    train, validation = [], []
    for cat in train_ids:
        candidates = sorted(by_id[cat], key=lambda r: r["sample_id"])
        if len(candidates) < 2:
            raise ValueError(f"training identity {cat!r} needs at least two images")
        chosen = candidates[rng.randrange(len(candidates))]
        validation.append(_protocol_row(chosen))
        train.extend(_protocol_row(r) for r in candidates if r is not chosen)

    def make_holdout(pool: list[str], name: str) -> tuple[dict, set[str]]:
        eligible = sorted(cat for cat in pool if len(by_id[cat]) >= enrollment + 1)
        if len(eligible) < gallery_size:
            raise ValueError(f"{name} holdout has no identity with at least enrollment+1 images")
        selector = random.Random(seed + (17 if name == "development" else 31))
        selector.shuffle(eligible)
        gallery_ids = sorted(eligible[:gallery_size])
        gallery, known = [], []
        used = set()
        for cat in gallery_ids:
            candidates = sorted(by_id[cat], key=lambda r: r["sample_id"])
            local = random.Random(seed + (17 if name == "development" else 31) + sum(ord(c) for c in cat))
            local.shuffle(candidates)
            gallery.extend(_protocol_row(r) for r in candidates[:enrollment])
            known.extend(_protocol_row(r) for r in candidates[enrollment:])
            used.add(cat)
        strangers = [cat for cat in pool if cat not in used]
        if len(strangers) < 2:
            raise ValueError(f"{name} needs at least two non-gallery identities for calibration/test unknown")
        strangers.sort()
        random.Random(seed + (101 if name == "development" else 211)).shuffle(strangers)
        midpoint = len(strangers) // 2
        if midpoint == 0 or midpoint == len(strangers):
            raise ValueError(f"{name} cannot split unknown identities")
        calibration_ids, unknown_ids = strangers[:midpoint], strangers[midpoint:]
        calibration = [_protocol_row(r) for cat in calibration_ids for r in sorted(by_id[cat], key=lambda x: x["sample_id"])]
        unknown = [_protocol_row(r) for cat in unknown_ids for r in sorted(by_id[cat], key=lambda x: x["sample_id"])]
        return {"gallery": gallery, "known": known, "calibration_unknown": calibration, "unknown": unknown}, used | set(calibration_ids) | set(unknown_ids)

    development, dev_used = make_holdout(dev_pool, "development")
    test, test_used = make_holdout(test_pool, "test")
    assigned = {r["sample_id"] for r in train + validation}
    for part in (development, test):
        for values in part.values():
            assigned.update(r["sample_id"] for r in values)
    if len(assigned) != len(rows):
        raise ValueError("protocol does not account for every clean input")
    try:
        manifest_name = Path(os.path.relpath(manifest.resolve(), Path.cwd().resolve())).as_posix()
    except ValueError:
        manifest_name = manifest.as_posix()
    return {
        "schema_version": 1, "seed": seed, "manifest": manifest_name,
        "diagnostic_only": True,
        "limitations": "Image-level diagnostic only; no session metadata or real-visit estimate.",
        "train": train, "validation": validation,
        "development": development, "test": test,
        "counts": {"total": len(rows), "assigned": len(assigned),
                    "train": len(train), "validation": len(validation),
                    "development": {k: len(v) for k, v in development.items()},
                    "test": {k: len(v) for k, v in test.items()}},
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("each embedding must be a finite, non-empty 1-D vector")
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("embedding must have non-zero norm")
    return value / norm


def predictions_from_embeddings(split: dict, embeddings: dict[str, np.ndarray], aggregation: str = "prototype") -> list[dict]:
    if aggregation != "prototype":
        raise ValueError("only prototype aggregation is supported")
    gallery = split.get("gallery", [])
    if not gallery:
        raise ValueError("split needs a non-empty gallery")
    prototypes = {}
    for row in gallery:
        if row["sample_id"] not in embeddings:
            raise ValueError(f"missing embedding for {row['sample_id']}")
        prototypes.setdefault(row["cat_id"], []).append(_unit(embeddings[row["sample_id"]]))
    prototypes = {cat: _unit(np.mean(values, axis=0)) for cat, values in prototypes.items()}
    predictions = []
    for split_name in ("known", "calibration_unknown", "unknown"):
        for row in split.get(split_name, []):
            if row["sample_id"] not in embeddings:
                raise ValueError(f"missing embedding for {row['sample_id']}")
            probe = _unit(embeddings[row["sample_id"]])
            cat, score = max(((cat, float(np.dot(probe, proto))) for cat, proto in prototypes.items()), key=lambda item: (item[1], item[0]))
            predictions.append({"sample_id": row["sample_id"], "split": split_name,
                                "cat_id": row["cat_id"], "predicted_id": cat, "score": score})
    return predictions


def evaluate_embeddings(protocol: dict, embeddings: dict[str, np.ndarray], partition: str = "development",
                        target_fpir: float = 0.01, aggregation: str = "prototype") -> tuple[list[dict], dict]:
    if partition not in ("development", "test"):
        raise ValueError("partition must be development or test")
    predictions = predictions_from_embeddings(protocol[partition], embeddings, aggregation)
    from catid.evaluation import evaluate
    return predictions, evaluate(predictions, target_fpir=target_fpir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gallery-size", type=int, default=5)
    parser.add_argument("--enrollment", type=int, default=3)
    args = parser.parse_args()
    protocol = build_protocol(args.manifest, gallery_size=args.gallery_size, enrollment=args.enrollment)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
