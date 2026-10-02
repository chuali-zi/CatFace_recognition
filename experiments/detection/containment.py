"""Optional research-only suppression for duplicate contained detections."""
from __future__ import annotations

import numpy as np


def suppress_contained(boxes, scores, threshold=0.9):
    """Return original indices after greedy containment suppression.

    A lower-scoring candidate is suppressed when its intersection with any
    already retained box divided by the candidate/retained minimum area is at
    least ``threshold``.  Coordinates are never merged or changed.
    """
    if not np.isfinite(threshold) or not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    values = np.asarray(boxes, dtype=np.float64)
    confidence = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4)")
        if confidence.size != 0:
            raise ValueError("boxes and scores must have the same length")
        return np.empty(0, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if confidence.ndim != 1 or confidence.shape[0] != values.shape[0]:
        raise ValueError("scores must have shape (N,)")
    if not np.isfinite(values).all() or not np.isfinite(confidence).all():
        raise ValueError("boxes and scores must be finite")
    widths = values[:, 2] - values[:, 0]; heights = values[:, 3] - values[:, 1]
    areas = widths * heights
    if np.any(widths <= 0) or np.any(heights <= 0) or not np.isfinite(areas).all():
        raise ValueError("boxes must have positive finite area")
    order = np.argsort(-confidence, kind="mergesort")
    kept = []
    for index in order:
        index = int(index)
        suppress = False
        for retained in kept:
            left = max(values[index, 0], values[retained, 0]); top = max(values[index, 1], values[retained, 1])
            right = min(values[index, 2], values[retained, 2]); bottom = min(values[index, 3], values[retained, 3])
            intersection = max(0.0, right - left) * max(0.0, bottom - top)
            if intersection / min(areas[index], areas[retained]) >= threshold:
                suppress = True
                break
        if not suppress:
            kept.append(index)
    return np.asarray(kept, dtype=np.int64)


def configure(model, out):
    """Copy the frozen model with a separate opt-in research sidecar."""
    import json
    import shutil
    from pathlib import Path
    from experiments.identity.runtime import model_digest

    model, out = Path(model), Path(out)
    metadata = json.loads(model.with_suffix(model.suffix + ".json").read_text(encoding="utf-8"))
    if metadata.get("model_sha256") != model_digest(model):
        raise ValueError("model and source metadata do not match")
    if (metadata.get("score_threshold") != 0.7 or metadata.get("nms_threshold") != 0.55 or
            metadata.get("containment_threshold") is not None):
        raise ValueError("use the original score0.7/NMS0.55 detector as the source")
    if out.exists() and any(out.iterdir()):
        raise ValueError("candidate directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    metadata["containment_threshold"] = 0.9
    metadata["postprocess"] += "; then greedy higher-score containment suppression at intersection/min(area)>=0.9"
    metadata["research_only"] = True
    # Torch parity was checked for the original postprocess, not this candidate.
    if "validation" in metadata:
        metadata["base_export_validation"] = metadata.pop("validation")
    metadata["containment_scope"] = "Fixed train/development geometric diagnostic; not field validated. Rebuild enrollment/calibration."
    shutil.copyfile(model, out / "model.onnx")
    (out / "model.onnx.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote research detector {out}; original model/configuration unchanged")


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Create a separate research detector with fixed containment threshold 0.9")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configure(args.model, args.out)


if __name__ == "__main__":
    main()
