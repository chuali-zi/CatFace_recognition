"""Repeated image-level gallery sensitivity diagnostics.

Episodes deliberately reuse the identities and images in an existing protocol's
development partition.  They are diagnostic resamplings, not additional cats
or independent visits.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Iterable


_GROUPS = ("gallery", "known", "calibration_unknown", "unknown")


def _rows(value: object, name: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"protocol {name} must be a list of row objects")
    return value


def _identity_set(rows: Iterable[dict], name: str) -> set[str]:
    result = set()
    for row in rows:
        cat = row.get("cat_id")
        if not isinstance(cat, str) or not cat:
            raise ValueError(f"{name} row must have a non-empty cat_id")
        result.add(cat)
    return result


def _development_pool(protocol: dict, partition: str) -> list[dict]:
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    split = protocol.get(partition)
    if not isinstance(split, dict):
        raise ValueError(f"protocol has no partition: {partition}")

    pool: list[dict] = []
    seen_sample: dict[str, dict] = {}
    seen_path: dict[str, dict] = {}
    for group in _GROUPS:
        for source in _rows(split.get(group, []), f"{partition}.{group}"):
            row = copy.deepcopy(source)
            sample_id = row.get("sample_id")
            path = row.get("path")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{partition}.{group} row must have a non-empty sample_id")
            if not isinstance(path, str) or not path:
                raise ValueError(f"{partition}.{group} row must have a non-empty path")
            if sample_id in seen_sample:
                raise ValueError(f"duplicate sample_id in {partition}: {sample_id}")
            if path in seen_path:
                raise ValueError(f"duplicate path in {partition}: {path}")
            seen_sample[sample_id] = row
            seen_path[path] = row
            pool.append(row)

    # Training/validation identities must never be drawn into the development
    # resampling pool.  This also catches malformed protocols early.
    train_ids = _identity_set(_rows(protocol.get("train", []), "train"), "train")
    validation_ids = _identity_set(_rows(protocol.get("validation", []), "validation"), "validation")
    pool_ids = _identity_set(pool, partition)
    overlap = (train_ids | validation_ids) & pool_ids
    if overlap:
        raise ValueError(f"training/validation identities overlap {partition}: {sorted(overlap)}")

    if isinstance(protocol.get("development"), dict) and isinstance(protocol.get("test"), dict):
        development_rows = [
            row for group in _GROUPS
            for row in _rows(protocol["development"].get(group, []), "development")
        ]
        test_rows = [row for group in _GROUPS for row in _rows(protocol["test"].get(group, []), "test")]
        overlap = _identity_set(development_rows, "development") & _identity_set(test_rows, "test")
        if overlap:
            raise ValueError(f"development/test identities overlap: {sorted(overlap)}")
    if not pool:
        raise ValueError(f"{partition} partition is empty")
    return pool


def build_episode(protocol: dict, partition: str = "development", seed: int = 0,
                  gallery_size: int = 5, enrollment: int = 3) -> dict:
    """Build one deterministic gallery/enrollment resampling of *partition*."""
    if gallery_size < 1 or enrollment < 1:
        raise ValueError("gallery_size and enrollment must be positive")
    pool = _development_pool(protocol, partition)
    by_id: dict[str, list[dict]] = {}
    for row in pool:
        by_id.setdefault(row["cat_id"], []).append(row)
    eligible = sorted(cat for cat, rows in by_id.items() if len(rows) >= enrollment + 1)
    if len(eligible) < gallery_size:
        raise ValueError(
            f"{partition} has {len(eligible)} eligible identities for enrollment={enrollment}; "
            f"need at least gallery_size={gallery_size}"
        )

    selector = random.Random(seed)
    selector.shuffle(eligible)
    gallery_ids = set(eligible[:gallery_size])
    strangers = sorted(cat for cat in by_id if cat not in gallery_ids)
    if len(strangers) < 2:
        raise ValueError("need at least two non-gallery identities for calibration/test unknown")
    stranger_rng = random.Random(seed + 1)
    stranger_rng.shuffle(strangers)
    midpoint = len(strangers) // 2
    if midpoint == 0 or midpoint == len(strangers):
        raise ValueError("cannot split unknown identities into calibration_unknown and unknown")
    calibration_ids = set(strangers[:midpoint])
    unknown_ids = set(strangers[midpoint:])

    result = {group: [] for group in _GROUPS}
    for cat in sorted(by_id):
        rows = sorted(by_id[cat], key=lambda row: (row["sample_id"], row["path"]))
        if cat in gallery_ids:
            local = random.Random(seed + 2 + sum(ord(char) for char in cat))
            local.shuffle(rows)
            result["gallery"].extend(rows[:enrollment])
            result["known"].extend(rows[enrollment:])
        elif cat in calibration_ids:
            result["calibration_unknown"].extend(rows)
        elif cat in unknown_ids:
            result["unknown"].extend(rows)
        else:  # defensive: every identity is assigned exactly once
            raise ValueError(f"identity was not assigned: {cat}")
    return result


def build_suite(protocol: dict, partition: str = "development", seeds: Iterable[int] = range(10),
                gallery_sizes: Iterable[int] = (2, 5), enrollments: Iterable[int] = (3, 5)) -> list[dict]:
    seeds = tuple(seeds)
    gallery_sizes = tuple(gallery_sizes)
    enrollments = tuple(enrollments)
    entries = []
    for seed in seeds:
        for gallery_size in gallery_sizes:
            for enrollment in enrollments:
                split = build_episode(protocol, partition, int(seed), int(gallery_size), int(enrollment))
                entries.append({
                    "episode_id": f"{partition}-seed{int(seed)}-gallery{int(gallery_size)}-enrollment{int(enrollment)}",
                    "seed": int(seed), "gallery_size": int(gallery_size),
                    "enrollment": int(enrollment), "split": split,
                })
    return entries


def build_paired_suite(protocol: dict, partition: str = "development", seeds: Iterable[int] = range(10),
                       gallery_sizes: Iterable[int] = (2, 5), enrollments: Iterable[int] = (3, 5)) -> list[dict]:
    """Build enrollment-paired episodes without changing the original sampler.

    For each seed and gallery size, a five-image enrollment is sampled once.
    Smaller enrollments use a prefix of each same gallery identity's sampled
    registration rows, while all post-registration known and unknown queries
    remain identical.  This isolates the effect of adding registration images
    within the fixed image-level diagnostic pool.
    """
    seeds = tuple(seeds)
    gallery_sizes = tuple(gallery_sizes)
    enrollments = tuple(enrollments)
    if not enrollments or any(int(value) < 1 for value in enrollments):
        raise ValueError("enrollments must contain positive values")
    # The paired design is explicitly anchored on five registration images;
    # every requested enrollment is a prefix of that same five-image draw.
    max_enrollment = 5
    entries = []
    for seed in seeds:
        for gallery_size in gallery_sizes:
            base = build_episode(protocol, partition, int(seed), int(gallery_size), max_enrollment)
            gallery_by_cat: dict[str, list[dict]] = {}
            for row in base["gallery"]:
                gallery_by_cat.setdefault(row["cat_id"], []).append(row)
            if any(len(rows) != max_enrollment for rows in gallery_by_cat.values()):
                raise ValueError("paired base episode does not contain max enrollment rows per gallery identity")
            for enrollment in enrollments:
                enrollment = int(enrollment)
                if enrollment > max_enrollment:
                    raise ValueError("enrollment cannot exceed paired maximum enrollment of 5")
                split = {
                    "gallery": [row for cat in sorted(gallery_by_cat) for row in gallery_by_cat[cat][:enrollment]],
                    "known": list(base["known"]),
                    "calibration_unknown": list(base["calibration_unknown"]),
                    "unknown": list(base["unknown"]),
                }
                unused = [row["sample_id"] for cat in sorted(gallery_by_cat)
                          for row in gallery_by_cat[cat][enrollment:]]
                entries.append({
                    "episode_id": f"{partition}-paired-seed{int(seed)}-gallery{int(gallery_size)}-enrollment{enrollment}",
                    "seed": int(seed), "gallery_size": int(gallery_size),
                    "enrollment": enrollment, "split": split,
                    "unused_enrollment_sample_ids": unused,
                })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partition", default="development")
    parser.add_argument("--seeds", type=int, default=10, help="generate seeds 0 through N-1")
    args = parser.parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    episodes = build_suite(protocol, partition=args.partition, seeds=range(args.seeds))
    output = {
        "diagnostic_only": True,
        "repeated_images": True,
        "selection_partition": args.partition,
        "episodes": episodes,
        "note": "Repeated image-level resampling within one identity pool; not independent cats or events and must not be added to sample-size denominators.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
