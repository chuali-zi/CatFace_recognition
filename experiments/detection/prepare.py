"""Prepare a non-Haar-selected CatFLW image-level detection experiment."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from .evaluate import iou


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def group_and_split(rows, seed=20260922):
    """Exact pixels are merged; conservative dHash components stay in one split.

    dHash is only a split precaution, not evidence of identity or a real session.
    Filename prefixes are NOT used: identical prefixes can show different cats.
    """
    by_pixel = defaultdict(list)
    for row in rows:
        by_pixel[row["pixel_hash"]].append(row)
    pixels = sorted(by_pixel)
    parent = list(range(len(pixels)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    signatures = [(int(by_pixel[p][0]["dhash"], 16), int(by_pixel[p][0]["dhash_mirror"], 16)) for p in pixels]
    for i, (a, am) in enumerate(signatures):
        for j in range(i):
            b, bm = signatures[j]
            if min((a ^ b).bit_count(), (a ^ bm).bit_count(), (am ^ b).bit_count()) <= 4:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, pixel in enumerate(pixels):
        groups[find(i)].append(pixel)
    components = sorted(groups.values(), key=lambda group: group[0])
    random.Random(seed).shuffle(components)
    result = []
    for index, group in enumerate(components):
        split = "train" if index < int(.70 * len(components)) else (
            "development" if index < int(.85 * len(components)) else "test")
        for pixel in group:
            duplicates = sorted(by_pixel[pixel], key=lambda row: row["path"])
            original = duplicates[0]
            boxes = []
            for duplicate in duplicates:
                box = duplicate["bounding_box"]
                # Duplicate files can annotate the same face with slightly different boxes.
                if not any(iou(box, old) >= .75 for old in boxes):
                    boxes.append(box)
            result.append({"sample_id": "catflw:" + pixel[:20], "path": original["path"],
                           "boxes": boxes, "split": split, "source": "catflw",
                           "width": original["width"], "height": original["height"],
                           "pixel_hash": pixel, "group_id": "dhash:" + group[0][:20],
                           "annotation_scope": "single_target", "review_status": "publisher_unreviewed",
                           "duplicate_paths": [row["path"] for row in duplicates]})
    return sorted(result, key=lambda row: row["sample_id"])


def review_sheets(rows, data_root, out, count=64, seed=20260922):
    """Choose review images before inference; sheets show author boxes, not predictions."""
    selected = []
    for split in ("development", "test"):
        pool = [row for row in rows if row["split"] == split]
        random.Random(f"{seed}:{split}:review").shuffle(pool)
        selected.extend(pool[:count])
    for index, row in enumerate(selected):
        row = dict(row, review_index=index)
        selected[index] = row
    write_rows(out / "review-selection.jsonl", selected)
    for start in range(0, len(selected), 16):
        sheet = Image.new("RGB", (1120, 1200), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for cell, row in enumerate(selected[start:start + 16]):
            x, y = (cell % 4) * 280, (cell // 4) * 300
            with Image.open(data_root / row["path"]) as source:
                image = source.convert("RGB")
                scale = min(270 / image.width, 264 / image.height)
                image = image.resize((round(image.width * scale), round(image.height * scale)))
                annotation = ImageDraw.Draw(image)
                for box in row["boxes"]:
                    annotation.rectangle(tuple(round(v * scale) for v in box), outline="#00ff00", width=2)
                sheet.paste(image, (x + 5, y + 30))
            draw.text((x + 5, y + 3), f"{row['review_index']:03} {row['split']} {Path(row['path']).stem}", fill="black")
        sheet.save(out / f"review-{start // 16:02}.jpg", quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/prepared.audit.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    source_rows = [row for row in read_rows(args.audit) if row["source"] == "catflw"]
    if not source_rows:
        parser.error("no CatFLW rows")
    for row in source_rows:
        stat = (args.data_root / row["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["file_size"], row["file_mtime_ns"]):
            raise ValueError(f"stale pixel/dHash audit: {row['path']}; re-run catid prepare")
        x1, y1, x2, y2 = row["bounding_box"]
        if row["exif_orientation"] != 1 or not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
            raise ValueError(f"orientation/box requires review: {row['path']}")
    rows = group_and_split(source_rows, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows(args.out / "manifest.jsonl", rows)
    review_sheets(rows, args.data_root, args.out, seed=args.seed)
    groups = Counter(row["group_id"] for row in rows)
    summary = {"seed": args.seed, "input_records": len(source_rows), "unique_pixels": len(rows),
               "groups": len(groups), "largest_group": max(groups.values()),
               "split_images": dict(Counter(row["split"] for row in rows)),
               "grouping": "RGB pixel SHA256 merge, then direct/mirrored dHash Hamming <=4 connected components",
               "annotation": "publisher xyxy boxes; duplicate-file boxes IoU>=.75 retain first path's box; may not exhaust all cats in frame; 48 points not used",
               "limitations": "No identities, sessions, official split, exhaustive negatives, or guarantee of all near-duplicate isolation. No Haar-based selection.",
               "source": "https://github.com/martvelge/CatFLW", "license": "CC BY-NC 4.0; research data, not a commercial-deployment clearance"}
    (args.out / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
