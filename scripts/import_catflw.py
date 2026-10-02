"""Extract the official CatFLW archive and append its original 48-point labels."""

import argparse
import json
import zipfile
from pathlib import Path

from catid.data import load_samples, save_samples


def import_catflw(archive, manifest, output):
    if output.resolve() == manifest.resolve() or output.resolve().parent != manifest.resolve().parent:
        raise ValueError("Use a separate output manifest beside the input")
    root = manifest.parent
    destination = root / "raw" / "georgemartvel__catflw"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Invalid archive path: {entry.filename}")
        bundle.extractall(destination)
    labels = sorted(destination.rglob("labels/*.json"))
    if not labels:
        raise ValueError("CatFLW archive contains no labels/*.json")
    rows = load_samples(manifest)
    if any(row["source"] == "catflw" for row in rows):
        raise ValueError("Input already contains CatFLW; use the original manifest")
    for label in labels:
        annotation = json.loads(label.read_text(encoding="utf-8"))
        image = label.parent.parent / "images" / (label.stem + ".png")
        if not image.is_file():
            raise ValueError(f"Missing image for {label}")
        points = annotation["labels"]
        if len(points) != 48 or any(len(point) != 2 for point in points):
            raise ValueError(f"Unexpected landmarks in {label}")
        rows.append(dict(path=image.relative_to(root).as_posix(), source="catflw", task="landmarks",
                         cat_id=None, session_id=None, review_status="pending", landmarks=points,
                         bounding_box=annotation["bounding_boxes"],
                         annotation_path=label.relative_to(root).as_posix()))
    save_samples(output, rows)
    print(json.dumps({"added_catflw": len(labels), "total": len(rows), "manifest": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/samples.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/samples.expanded.jsonl"))
    args = parser.parse_args()
    import_catflw(args.archive, args.manifest, args.out)
