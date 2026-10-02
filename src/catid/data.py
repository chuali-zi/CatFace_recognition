"""Portable sample manifests and exact-pixel duplicate removal."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


def load_samples(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Line {number}: path must be relative to the manifest")
        if row["path"] in seen:
            raise ValueError(f"Line {number}: repeated sample path")
        seen.add(row["path"])
        if row["task"] not in {"identity", "landmarks"}:
            raise ValueError(f"Line {number}: unsupported task")
        identity = row.get("cat_id")
        if row["task"] == "identity" and (not isinstance(identity, str) or not identity.strip()):
            raise ValueError(f"Line {number}: identity samples require cat_id")
        if row["task"] == "landmarks" and identity is not None:
            raise ValueError(f"Line {number}: landmark-only samples have no identity")
        rows.append(row)
    return rows


def summarize(rows: list[dict], root: Path) -> dict:
    sources = {}
    for source in sorted({row["source"] for row in rows}):
        items = [row for row in rows if row["source"] == source]
        sources[source] = {
            "samples": len(items),
            "identities": len({row["cat_id"] for row in items if row["task"] == "identity"}),
            "tasks": dict(Counter(row["task"] for row in items)),
            "review_status": dict(Counter(row["review_status"] for row in items)),
            "without_session": sum(row.get("session_id") is None for row in items),
        }
    return {"samples": len(rows),
            "missing_files": sum(not (root / row["path"]).is_file() for row in rows),
            "sources": sources}


def clean_duplicates(rows: list[dict], root: Path, sources: set[str]) -> tuple[list[dict], dict]:
    """Exclude all conflicting identities, keeping the first same-ID duplicate.

    Images and human review status are never modified.
    """
    available = {row["source"] for row in rows if row["task"] == "identity"}
    if not sources or not sources <= available:
        raise ValueError(f"Choose identity sources from {sorted(available)}")
    groups = defaultdict(list)
    removed = {}
    examined = 0
    for row in rows:
        if row["source"] not in sources or row["task"] != "identity":
            continue
        examined += 1
        try:
            with Image.open(root / row["path"]) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                key = (image.size, hashlib.sha256(image.tobytes()).digest())
        except (OSError, UnidentifiedImageError):
            removed[row["path"]] = "unreadable"
            continue
        groups[key].append(row)
    for group in groups.values():
        if len({row["cat_id"] for row in group}) > 1:
            for row in group:
                removed[row["path"]] = "conflicting_identity"
        else:
            for row in group[1:]:
                removed[row["path"]] = "duplicate"
    cleaned = [row for row in rows if row["path"] not in removed]
    return cleaned, {
        "sources": sorted(sources), "examined": examined,
        "kept_in_scope": examined - len(removed),
        "removed_counts": dict(Counter(removed.values())), "removed": removed,
        "scope": "Exact EXIF-oriented RGB pixels only; no near-duplicate, session or manual identity validation.",
    }


def save_samples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
