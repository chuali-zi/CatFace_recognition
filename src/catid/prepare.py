"""Decode and prepare research data without changing originals or certifying labels."""

import hashlib
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps, ImageStat

from .data import load_samples, save_samples, summarize


def dhash(image):
    pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    return sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))


class HashIndex:
    """Small BK tree for Hamming-distance candidate search."""

    def __init__(self):
        self.root = None

    def add(self, key, index):
        if self.root is None:
            self.root = (key, [index], {})
            return
        node = self.root
        while True:
            distance = (node[0] ^ key).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = (key, [index], {})
                return
            node = node[2][distance]

    def find(self, key, radius):
        stack = [self.root] if self.root else []
        while stack:
            value, indices, children = stack.pop()
            distance = (value ^ key).bit_count()
            if distance <= radius:
                for index in indices:
                    yield index, distance
            stack.extend(child for edge, child in children.items()
                         if distance - radius <= edge <= distance + radius)


def scan(row, root):
    result = dict(row)
    result["preparation_flags"] = []
    stat = (root / row["path"]).stat()
    result.update(file_size=stat.st_size, file_mtime_ns=stat.st_mtime_ns, scan_version=1)
    with Image.open(root / row["path"]) as original:
        original.load()
        orientation = original.getexif().get(274, 1)
        image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        result.update(width=width, height=height, exif_orientation=orientation,
                      pixel_hash=hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest(),
                      dhash=f"{dhash(image):016x}",
                      dhash_mirror=f"{dhash(ImageOps.mirror(image)):016x}")
        if min(width, height) < 32:
            result["preparation_flags"].append("small_image")
        if max(ImageStat.Stat(image.resize((32, 32))).stddev) < 2:
            result["preparation_flags"].append("almost_constant")
        if row["task"] == "landmarks":
            points = row.get("landmarks")
            expected = {"catface": 9, "catflw": 48}.get(row["source"])
            valid = (isinstance(points, list) and len(points) > 0
                     and (expected is None or len(points) == expected)
                     and all(isinstance(p, (list, tuple)) and len(p) == 2
                             and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)
                             for p in points))
            if not valid or orientation != 1:
                result["preparation_flags"].append("landmark_format_or_orientation")
            else:
                result["landmark_in_frame"] = [0 <= x < width and 0 <= y < height for x, y in points]
                if not all(result["landmark_in_frame"]):
                    result["preparation_flags"].append("landmarks_outside_image")
            if "bounding_box" in row:
                box = row["bounding_box"]
                if not (isinstance(box, list) and len(box) == 4
                        and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                        and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    result["preparation_flags"].append("invalid_bounding_box")
    return result


def prepare(manifest: Path, radius=4):
    """Outputs stay beside the input so image paths remain portable.

    Near matches are candidates, never automatic identity merges. No artificial
    sessions or final train/test split are invented from unreviewed labels.
    """
    if not 0 <= radius <= 8:
        raise ValueError("near-distance must be between 0 and 8")
    root = manifest.parent
    if manifest.name.startswith("prepared."):
        raise ValueError("Use the original or expanded manifest, not a prepared output")
    rows = load_samples(manifest)
    scanned, rejected = [], []
    cache_path = root / "prepared.scan.jsonl"
    cache = {r["path"]: r for r in load_samples(cache_path)} if cache_path.exists() else {}
    def inspect(row):
        try:
            cached = cache.get(row["path"])
            stat = (root / row["path"]).stat()
            if (cached and cached.get("scan_version") == 1
                    and cached.get("file_size") == stat.st_size
                    and cached.get("file_mtime_ns") == stat.st_mtime_ns
                    and all(cached.get(key) == value for key, value in row.items())):
                return cached, None
            return scan(row, root), None
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            return None, dict(row, preparation_flags=["unreadable"], error=str(error))

    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, (good, bad) in enumerate(pool.map(inspect, rows), 1):
            if good is not None:
                scanned.append(good)
            else:
                rejected.append(bad)
            if number % 1000 == 0:
                checkpoint = cache_path.with_suffix(".tmp")
                save_samples(checkpoint, scanned)
                checkpoint.replace(cache_path)
                print(f"Decoded {number}/{len(rows)}", flush=True)
    save_samples(cache_path, scanned)

    groups = defaultdict(list)
    for row in scanned:
        groups[row["pixel_hash"]].append(row)
    kept, duplicates, conflicts = [], [], []
    for group in groups.values():
        identities = {r["cat_id"] for r in group if r["task"] == "identity"}
        for task in ("identity", "landmarks"):
            items = [r for r in group if r["task"] == task]
            if not items:
                continue
            if task == "identity" and len(identities) > 1:
                conflicts.extend(dict(r, exclusion_reason="exact_identity_conflict") for r in items)
                continue
            # Different landmark schemas on the same image are useful annotations.
            schemas = defaultdict(list)
            for row in items:
                schemas[row["source"] if task == "landmarks" else "identity"].append(row)
            for schema in schemas.values():
                if task == "landmarks" and len({json.dumps(r["landmarks"]) for r in schema}) > 1:
                    conflicts.extend(dict(r, exclusion_reason="exact_annotation_conflict") for r in schema)
                    continue
                kept.append(schema[0])
                duplicates.extend(dict(r, duplicate_of=schema[0]["path"]) for r in schema[1:])

    # Compare unique pixels globally, including cross-task and cross-source matches.
    index = HashIndex()
    representatives = [items[0] for items in groups.values()]
    edges, near_paths = [], set()
    for number, row in enumerate(representatives):
        matches = {}
        for field in ("dhash", "dhash_mirror"):
            for other, distance in index.find(int(row[field], 16), radius):
                matches[other] = min(matches.get(other, 65), distance)
        for other, distance in matches.items():
            previous = representatives[other]
            edges.append(dict(path_a=previous["path"], path_b=row["path"],
                              hash_a=previous["pixel_hash"], hash_b=row["pixel_hash"], distance=distance))
            near_paths.update((previous["pixel_hash"], row["pixel_hash"]))
        index.add(int(row["dhash"], 16), number)
        if number and number % 5000 == 0:
            print(f"Near-duplicate search {number}/{len(representatives)}", flush=True)

    identity, landmarks, review = [], [], []
    manual_path = root / "manual-review.csv"
    manual = {}
    if manual_path.exists():
        with manual_path.open(encoding="utf-8-sig", newline="") as stream:
            manual = {item["path"]: item["reason"] for item in csv.DictReader(stream)}
        unknown = set(manual) - {row["path"] for row in rows}
        if unknown:
            raise ValueError(f"Manual review paths not found in input: {sorted(unknown)[:3]}")
    manual_hashes = defaultdict(set)
    for row in scanned:
        if row["path"] in manual:
            manual_hashes[row["pixel_hash"]].add(manual[row["path"]])
    group_review = {}
    group_path = root / "manual-group-review.csv"
    if group_path.exists():
        with group_path.open(encoding="utf-8-sig", newline="") as stream:
            group_review = {item["cat_id"]: item["reason"] for item in csv.DictReader(stream)}
        unknown_groups = set(group_review) - {row["cat_id"] for row in rows if row["task"] == "identity"}
        if unknown_groups:
            raise ValueError(f"Group review identities not found in input: {sorted(unknown_groups)[:3]}")
    for row in kept:
        flags = row["preparation_flags"]
        flags.extend(sorted(manual_hashes[row["pixel_hash"]]))
        if row["task"] == "identity" and row["cat_id"] in group_review:
            flags.append(group_review[row["cat_id"]])
        if row["pixel_hash"] in near_paths:
            flags.append("near_duplicate_candidate")
        if row["source"] == "reid":
            flags.append("source_identity_unreliable")
        # Strict subsets: uncertain near matches and out-of-frame points remain
        # available in review, not silently clipped, relabelled or discarded.
        if flags:
            review.append(row)
        elif row["task"] == "identity":
            identity.append(row)
        else:
            landmarks.append(row)

    counts = Counter(row["cat_id"] for row in identity)
    singletons = [dict(row, exclusion_reason="fewer_than_two_clean_images")
                  for row in identity if counts[row["cat_id"]] < 2]
    identity = [row for row in identity if counts[row["cat_id"]] >= 2]
    outputs = {"identity": identity, "landmarks": landmarks, "review": review,
               "duplicates": duplicates, "conflicts": conflicts, "unreadable": rejected,
               "singletons": singletons, "audit": scanned, "near-pairs": edges}
    for name, items in outputs.items():
        save_samples(root / f"prepared.{name}.jsonl", items)
    report = {
        "input": manifest.name, "input_samples": len(rows), "near_distance": radius,
        "outputs": {name: len(items) for name, items in outputs.items()},
        "identity": summarize(identity, root), "landmarks": summarize(landmarks, root),
        "review_flags": dict(Counter(flag for row in review for flag in row["preparation_flags"])),
        "status": "Mechanically prepared research subsets; identity labels remain pending human review. No final evaluation split.",
        "near_method": "64-bit dHash Hamming distance, original and horizontal mirror; candidates only, not exhaustive.",
    }
    (root / "prepared.summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
