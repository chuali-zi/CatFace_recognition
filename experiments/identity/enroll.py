"""Build a calibrated, model-specific enrollment gallery from user images."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

from catid.evaluation import threshold_at_fpir
from .runtime import Encoder, embed_rows, make_gallery, match_gallery, model_digest


def build_calibrated_gallery(rows, embeddings, model_metadata, target_fpir=0.01):
    embedding_dim = int(model_metadata["embedding_dim"])
    for sample_id, vector in embeddings.items():
        value = np.asarray(vector)
        if value.ndim != 1 or value.shape[0] != embedding_dim:
            raise ValueError(f"Embedding dimension mismatch for {sample_id}: expected {embedding_dim}")
    gallery_rows = [r for r in rows if r["split"] == "gallery"]
    unknown_rows = [r for r in rows if r["split"] == "calibration_unknown"]
    if not gallery_rows or not unknown_rows:
        raise ValueError("manifest needs non-empty gallery and calibration_unknown")
    gallery_ids = {r["cat_id"] for r in gallery_rows}
    unknown_ids = {r["cat_id"] for r in unknown_rows}
    if gallery_ids & unknown_ids: raise ValueError("gallery and calibration_unknown identities must be disjoint")
    identities, vectors = make_gallery(gallery_rows, embeddings)
    gallery = {"cat_ids": identities, "vectors": [v.tolist() for v in vectors]}
    scores = [match_gallery(embeddings[r["sample_id"]], gallery)[1] for r in unknown_rows]
    threshold = threshold_at_fpir(scores, target_fpir)
    result = {"model_sha256": model_metadata["model_sha256"], "input_size": int(model_metadata["input_size"]), "embedding_dim": embedding_dim, "aggregation": "prototype", "cat_ids": identities, "vectors": [v.tolist() for v in vectors], "threshold": threshold, "diagnostic_only": True, "n_gallery_images": len(gallery_rows), "n_gallery_identities": len(identities), "n_calibration_unknown": len(unknown_rows), "target_calibration_fpir": target_fpir, "note": "Image-level threshold calibrated for this exact enrollment gallery and model; recalibrate after changes."}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--target-fpir", type=float, default=0.01); p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    if args.threads < 1: raise ValueError("--threads must be positive")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows: raise ValueError("manifest is empty")
    seen_paths, seen_pixels = set(), set()
    for row in rows:
        if row.get("split") not in {"gallery", "calibration_unknown"}: raise ValueError("split must be gallery or calibration_unknown")
        if not isinstance(row.get("cat_id"), str) or not row["cat_id"].strip(): raise ValueError("cat_id must be nonempty")
        path = Path(row.get("path", "")); resolved = path if path.is_absolute() else args.manifest.parent / path
        if not path.name or not resolved.is_file(): raise ValueError(f"image path does not exist: {path}")
        key = str(resolved.resolve())
        if key in seen_paths: raise ValueError("sample paths must be unique")
        seen_paths.add(key)
        with Image.open(resolved) as source:
            pixels = ImageOps.exif_transpose(source).convert("RGB")
            digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
        if digest in seen_pixels: raise ValueError("duplicate image pixels across enrollment rows")
        seen_pixels.add(digest)
        row["sample_id"] = key
    encoder = Encoder(args.model, threads=args.threads)
    embeddings = embed_rows(encoder, rows, args.manifest.parent)
    result = build_calibrated_gallery(rows, embeddings, {"model_sha256": model_digest(args.model), "input_size": encoder.size, "embedding_dim": int(encoder.metadata["embedding_dim"])}, args.target_fpir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"vectors", "model_sha256"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
