"""Prepare human-review candidates from quarantined re-ID advertising/listing images.

This reads existing audit records and pixels only. It does not infer ``no_cat``
with a model; every candidate remains unverified until a person reviews it.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def load_candidates(audit: Path) -> list[dict]:
    candidates = []
    seen = set()
    with audit.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            flags = set(row.get("preparation_flags", []))
            if row.get("source") != "reid" or "source_identity_unreliable" not in flags:
                continue
            pixel_hash = row.get("pixel_hash")
            if not pixel_hash or pixel_hash in seen:
                continue
            seen.add(pixel_hash)
            candidates.append({"path": row["path"], "source": row["source"],
                               "pixel_hash": pixel_hash, "preparation_flags": sorted(flags),
                               "cat_id": row.get("cat_id"), "session_id": row.get("session_id"),
                               "review_status": row.get("review_status")})
    return sorted(candidates, key=lambda row: row["pixel_hash"])


def make_contact_sheet(rows: list[dict], root: Path, output: Path) -> None:
    columns, cell_w, cell_h = 4, 320, 280
    sheet = Image.new("RGB", (columns * cell_w, ((len(rows) + columns - 1) // columns) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        path = root / row["path"]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((cell_w - 12, cell_h - 42))
            x = (index % columns) * cell_w + (cell_w - image.width) // 2
            y = (index // columns) * cell_h + 4
            sheet.paste(image, (x, y))
        label = f"{index:02d}  {Path(row['path']).name}"
        draw.text(((index % columns) * cell_w + 4, (index // columns) * cell_h + cell_h - 32), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/prepared.audit.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    candidates = load_candidates(args.audit)
    selected = random.Random(args.seed).sample(candidates, min(args.limit, len(candidates)))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    make_contact_sheet(selected, args.data_root, args.out / "contact-sheet.png")
    report = {
        "diagnostic_only": True, "verified_no_cat": False,
        "purpose": "Human-review candidates for an advertising/listing-derived negative challenge; not door no-cat FPR.",
        "audit": str(args.audit), "seed": args.seed, "candidate_limit": args.limit,
        "raw_reid_flagged_rows": len(candidates), "raw_reid_unique_pixel_images": len(candidates),
        "selected": len(selected), "source_counts": dict(Counter(row["source"] for row in candidates)),
        "flag_counts": dict(Counter(flag for row in candidates for flag in row["preparation_flags"])),
        "selection": "fixed-seed sample after pixel_hash deduplication; no model or filename identity inference",
        "review_required": "Review every contact-sheet tile for actual absence of cats before using any row as a negative.",
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
