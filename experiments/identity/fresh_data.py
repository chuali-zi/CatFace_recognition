"""Select unexposed publisher identities and render frozen-detector review sheets.

Selection uses IDs and original-image hashes, never recognition predictions.
Preparation creates pending candidates; explicit reviews can freeze a separate
diagnostic protocol. Neither operation changes the clean dataset.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from experiments.detection.onnx import Detector, single_crop
from .runtime import model_digest, read_image

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def hash_distance(a, b):
    return min((int(a["dhash"], 16) ^ int(b[key], 16)).bit_count()
               for key in ("dhash", "dhash_mirror"))


def select_candidates(rows, old_protocol, old_original_hashes, identities=40, per_identity=8, seed=20260926):
    old_ids = {r["cat_id"] for key in ("train", "validation") for r in old_protocol[key]}
    old_ids.update(r["cat_id"] for key in ("development", "test")
                   for group in old_protocol[key].values() for r in group)
    groups = defaultdict(list)
    hashes = set()
    for row in sorted(rows, key=lambda r: r["path"]):
        if row["source"] != "individuals" or row["cat_id"] in old_ids:
            continue
        pixel = row["pixel_hash"]
        if pixel in old_original_hashes:
            raise ValueError("new publisher identity duplicates an old original image")
        if pixel in hashes:
            raise ValueError("candidate pool contains duplicate pixels; resolve labels first")
        hashes.add(pixel)
        groups[row["cat_id"]].append(row)
    eligible = sorted(cat for cat, values in groups.items() if len(values) >= per_identity)
    if identities < 1 or per_identity < 1 or len(eligible) < identities:
        raise ValueError("not enough eligible new identities")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = []
    for cat in sorted(eligible[:identities]):
        remaining = list(groups[cat])
        chosen = [remaining.pop(rng.randrange(len(remaining)))]
        while remaining and len(chosen) < per_identity:
            candidate = max(remaining, key=lambda r: (min(hash_distance(r, c) for c in chosen), r["path"]))
            chosen.append(candidate)
            remaining.remove(candidate)
        for rank, row in enumerate(chosen):
            selected.append({**row, "selection_rank": rank,
                             "min_prior_dhash_distance": min(hash_distance(row, c) for c in chosen[:rank]) if rank else None})
    return sorted(selected, key=lambda r: (r["cat_id"], r["path"])), {
        "new_pool_identities": len(groups), "new_pool_images": sum(map(len, groups.values())),
        "eligible_identities": len(eligible), "selected_identities": identities,
        "per_identity": per_identity, "seed": seed,
        "method": "seeded identity selection; original/mirrored dHash max-min images; no detector or identity scores"}


def review_sheet(cat_id, frames, destination):
    cell_w, cell_h, columns = 420, 266, 4
    sheet = Image.new("RGB", (cell_w * columns, 44 + cell_h * ((len(frames) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 10), f"{cat_id} | LEFT original / RIGHT detector crop | PENDING visual review", fill="black", font=font)
    for index, (image, crop, row) in enumerate(frames):
        x, y = (index % columns) * cell_w, 44 + (index // columns) * cell_h
        draw.text((x + 4, y + 2), f"{row['review_index']:03d} {Path(row['path']).name}", fill="black", font=font)
        original = image.copy()
        box_draw = ImageDraw.Draw(original)
        for box in row["boxes"]:
            box_draw.rectangle(tuple(box), outline="red", width=max(2, image.width // 300))
        for offset, picture in ((4, original), (198, crop)):
            if picture is not None:
                thumb = ImageOps.contain(picture, (188 if offset == 4 else 216, 216))
                sheet.paste(thumb, (x + offset, y + 26 + (216 - thumb.height) // 2))
        draw.text((x + 4, y + 243), f"{row['status']} | crop {row['crop_size']} | dH {row['min_prior_dhash_distance']}", fill="black", font=font)
    sheet.save(destination, quality=94)


def prepare(out, identities=40, per_identity=8, seed=20260926, threads=2):
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be empty or new")
    protocol = json.loads((ROOT / "experiments/runs/edge160/protocol.json").read_text(encoding="utf-8"))
    clean = read_rows(ROOT / "data/clean.identity.jsonl")
    rows, selection = select_candidates(read_rows(ROOT / "data/prepared.identity.jsonl"), protocol,
                                         {r["original_pixel_hash"] for r in clean}, identities, per_identity, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(exist_ok=True)
    model = ROOT / "experiments/runs/detection-cpu/onnx/model.onnx"
    detector = Detector(model, threads)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["cat_id"]].append((index, row))
    observations = []
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for cat, items in groups.items():
            frames = []
            for index, row in items:
                path = ROOT / "data" / row["path"]
                if path.stat().st_size != row["file_size"]:
                    raise ValueError(f"source image changed: {path}")
                image = read_image(path)
                boxes, scores = detector.detect(image)
                status, box, crop = single_crop(image, boxes, "head")
                record = {**row, "sample_id": row["path"], "review_index": index,
                          "review_status": "pending", "boxes": boxes.tolist(), "scores": scores.tolist(),
                          "status": status, "crop_box": box, "crop_size": list(crop.size) if crop is not None else None}
                observations.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                frames.append((image, crop, record))
            stream.flush()
            review_sheet(cat, frames, out / "review" / f"{cat.split(':')[-1]}.jpg")
            print(f"review sheet {cat}: {len(observations)}/{len(rows)}", flush=True)
    metadata = {**selection, "n_images": len(rows), "status_counts": dict(Counter(r["status"] for r in observations)),
                "detector_sha256": model_digest(model), "detector_metadata_sha256": detector.metadata_sha256,
                "crop_mode": "head", "identity_inference_run": False,
                "scope": "Pending visual/identity/duplicate review; no field sessions; no automatic admission to clean"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def merge_reviews(candidates, reviews, quarantine_ids=()):
    """Attach explicit visual decisions; never infer a review from detection."""
    by_index = {r["review_index"]: r for r in candidates}
    if len(by_index) != len(candidates):
        raise ValueError("duplicate candidate review_index")
    decisions = {}
    seen_groups = set()
    quarantine_ids = set(quarantine_ids)
    for review in reviews:
        for group in review["groups"]:
            cat = group["cat_id"]
            if cat in seen_groups:
                raise ValueError("duplicate reviewed group")
            seen_groups.add(cat)
            if group["group_decision"] not in {"appearance_consistent", "identity_uncertain", "mixed"}:
                raise ValueError("invalid group decision")
            expected = {i for i, r in by_index.items() if r["cat_id"] == cat}
            actual = [r["review_index"] for r in group["images"]]
            if not expected or set(actual) != expected or len(set(actual)) != len(actual):
                raise ValueError("review group indices do not match candidates")
            for decision in group["images"]:
                index = decision["review_index"]
                row = by_index[index]
                if decision.get("sample_id", row["sample_id"]) != row["sample_id"]:
                    raise ValueError("review sample_id mismatch")
                if decision["visual"] not in {"usable", "difficult", "unusable", "uncertain"} or not decision["reason"]:
                    raise ValueError("invalid or unexplained visual decision")
                reason = None
                if cat in quarantine_ids:
                    reason = "whole identity quarantined: unresolved possible same cat under an old publisher label"
                elif group["group_decision"] != "appearance_consistent":
                    reason = "group identity appearance requires further review"
                elif decision["visual"] in {"unusable", "uncertain"}:
                    reason = "visual target/identity cannot be established"
                decisions[index] = {**row, "review_status": "visually_reviewed",
                                    "group_decision": group["group_decision"], "group_note": group["group_note"],
                                    "visual": decision["visual"], "visual_reason": decision["reason"],
                                    "evaluation_include": reason is None, "exclusion_reason": reason}
    if set(decisions) != set(by_index):
        raise ValueError("missing visual reviews")
    if quarantine_ids - seen_groups:
        raise ValueError("quarantine identity absent from candidate review")
    return [decisions[i] for i in sorted(decisions)]


def build_reviewed_episodes(rows, seed=20260926, count=10, gallery_size=5, enrollment=3):
    """Quality-reviewed enrollment only; all remaining included images queried.

    Human-readable enrollment is a collection constraint, not a query filter.
    Roles are frozen before identity inference; repeated seeds are diagnostics.
    """
    if min(count, gallery_size, enrollment) < 1:
        raise ValueError("episode sizes must be positive")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        if row["evaluation_include"]:
            groups[row["cat_id"]].append(row)
    good = {cat: [r for r in values if r["visual"] == "usable" and r["status"] == "ok"]
            for cat, values in groups.items()}
    eligible = sorted(cat for cat, values in good.items()
                      if len(values) >= enrollment and len(groups[cat]) > enrollment)
    if len(eligible) < gallery_size or len(groups) - gallery_size < 2:
        raise ValueError("insufficient reviewed enrollment or stranger identities")
    episodes = []
    for index in range(count):
        rng = random.Random(seed + index)
        gallery_ids = set(rng.sample(eligible, gallery_size))
        strangers = sorted(set(groups) - gallery_ids)
        rng.shuffle(strangers)
        calibration_ids = set(strangers[:len(strangers) // 2])
        split = {key: [] for key in ("gallery", "known", "calibration_unknown", "unknown")}
        for cat, values in sorted(groups.items()):
            enrollment_ids = {r["sample_id"] for r in rng.sample(good[cat], enrollment)} if cat in gallery_ids else set()
            for row in values:
                if cat in gallery_ids:
                    role = "gallery" if row["sample_id"] in enrollment_ids else "known"
                else:
                    role = "calibration_unknown" if cat in calibration_ids else "unknown"
                split[role].append({key: row[key] for key in ("sample_id", "cat_id", "path")})
        episodes.append({"episode_id": f"fresh-seed{seed + index}-gallery{gallery_size}-enrollment{enrollment}",
                         "seed": seed + index, "gallery_size": gallery_size, "enrollment": enrollment, "split": split})
    return episodes, eligible


def finalize(out, review_paths, seed=20260926):
    destinations = [out / name for name in ("reviewed.jsonl", "review-summary.json", "protocol.json")]
    if any(path.exists() for path in destinations):
        raise ValueError("review/protocol outputs already exist; do not silently replace frozen roles")
    duplicate_review = json.loads((out / "duplicate-review.json").read_text(encoding="utf-8"))
    quarantine = {p["selected_cat_id"] for p in duplicate_review["pairs"] if p["isolate"]}
    reviews = [json.loads(path.read_text(encoding="utf-8-sig")) for path in review_paths]
    reviewed = merge_reviews(read_rows(out / "candidates.jsonl"), reviews, quarantine)
    episodes, eligible = build_reviewed_episodes(reviewed, seed)
    included = [r for r in reviewed if r["evaluation_include"]]
    summary = {"reviewed_images": len(reviewed), "reviewed_identities": len({r["cat_id"] for r in reviewed}),
               "visual_counts": dict(Counter(r["visual"] for r in reviewed)),
               "included_images": len(included), "included_identities": len({r["cat_id"] for r in included}),
               "included_visual_counts": dict(Counter(r["visual"] for r in included)),
               "included_detector_status_counts": dict(Counter(r["status"] for r in included)),
               "quarantined_identities": sorted(quarantine), "gallery_eligible_identities": eligible,
               "scope": "Publisher labels with appearance review, not field-verified identity truth; no session independence; clean dataset unchanged"}
    protocol = {"seed": seed, "episodes": episodes, "identity_inference_at_freeze": False,
                "gallery_policy": "Choose five identities with >=3 visually usable single-head crops; enroll three seeded images. No identity scores used.",
                "query_policy": "All other included images retained, including difficult, no_cat and duplicate-box failures. Unknown calibration/query identities disjoint within episode.",
                "arms": ["small160_raw", "large224_raw", "large224_center", "large224_whiten128"],
                "target_fpir": 0.01, "transform_fit": "Existing old-train-only 128D transform; never refit on fresh identities",
                "scope": "Image-level confirmation of fixed candidates; ten resamplings are not ten independent datasets or visits"}
    destinations[0].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviewed), encoding="utf-8")
    for path, value in zip(destinations[1:], (summary, protocol)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identities", type=int, default=40)
    parser.add_argument("--per-identity", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--finalize", action="store_true", help="merge explicit reviews and freeze roles, without identity inference")
    parser.add_argument("--reviews", nargs="+", type=Path)
    args = parser.parse_args()
    if args.finalize:
        if not args.reviews:
            parser.error("--finalize requires --reviews")
        finalize(args.out, args.reviews, args.seed)
    else:
        prepare(args.out, args.identities, args.per_identity, args.seed, args.threads)


if __name__ == "__main__":
    main()
