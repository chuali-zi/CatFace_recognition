"""Small, fixed-shape SSDLite320 export and CPU ONNX inference helpers.

The exported graph deliberately exposes decoded 320x320 boxes and the COCO
cat (class 17) softmax probability.  Torch is imported only by ``export``;
prediction is NumPy/Pillow/ONNX Runtime only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from .containment import suppress_contained

IMAGE_SIZE = 320
CAT_LABEL = 17


def crop_policy(mode="head"):
    if mode not in {"head", "face_core"}:
        raise ValueError(f"unknown crop mode: {mode}")
    region = "whole head box" if mode == "head" else "head width/height * 2/3, center shifted down by head height * 0.10"
    return f"Exactly one score-filtered detector box; {region}; rounded xyxy original pixels; no reference-box selection or fallback"


def single_crop(image, boxes, crop_mode="head"):
    """One geometric policy shared by feature studies, enrollment and replay."""
    crop_policy(crop_mode)
    if len(boxes) == 0:
        return "no_cat", None, None
    if len(boxes) > 1:
        return "unusable", None, None
    values = [float(value) for value in boxes[0]]
    if crop_mode == "face_core":
        x1, y1, x2, y2 = values
        width, height = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + height * 0.10
        values = [cx - width / 3, cy - height / 3, cx + width / 3, cy + height / 3]
    box = [round(value) for value in values]
    if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
        raise ValueError(f"invalid detector crop: {box} for {image.size}")
    return "ok", box, image.crop(tuple(box))


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    keep = []
    while len(order):
        i = int(order[0]); keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0]); y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2]); y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        area = np.maximum(0, boxes[i, 2]-boxes[i, 0]) * np.maximum(0, boxes[i, 3]-boxes[i, 1])
        area_r = np.maximum(0, boxes[rest, 2]-boxes[rest, 0]) * np.maximum(0, boxes[rest, 3]-boxes[rest, 1])
        order = rest[inter / np.maximum(area + area_r - inter, 1e-12) <= threshold]
    return np.asarray(keep, dtype=np.int64)


def _preprocess(path: Path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    array = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None], (width, height)


def _select(boxes, scores, metadata):
    threshold = float(metadata["score_threshold"])
    nms_threshold = float(metadata["nms_threshold"])
    topk = int(metadata["topk_candidates"])
    limit = int(metadata["max_boxes"])
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    keep = np.flatnonzero(scores >= threshold)
    keep = keep[np.argsort(-scores[keep], kind="mergesort")[:topk]]
    keep = keep[_nms(boxes[keep], scores[keep], nms_threshold)]
    keep = keep[np.argsort(-scores[keep], kind="mergesort")[:limit]]
    boxes, scores = boxes[keep], scores[keep]
    containment = metadata.get("containment_threshold")
    if containment is not None:
        kept = suppress_contained(boxes, scores, float(containment))
        boxes, scores = boxes[kept], scores[kept]
    return boxes, scores


class Detector:
    """Reusable CPU detector for an already-decoded PIL RGB image."""

    def __init__(self, model, threads=2):
        import onnxruntime as ort
        if threads < 1:
            raise ValueError("threads must be positive")
        self.model = Path(model)
        metadata_path = self.model.with_suffix(self.model.suffix + ".json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model_sha256 = _sha256(self.model)
        self.metadata_sha256 = _sha256(metadata_path)
        if self.metadata.get("model_sha256", self.model_sha256) != self.model_sha256:
            raise ValueError("detector model does not match its metadata")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model), providers=["CPUExecutionProvider"], sess_options=options)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        array = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        boxes, scores = self.session.run(None, {self.input_name: np.transpose(array, (2, 0, 1))[None]})
        boxes, scores = _select(boxes, scores, self.metadata)
        scale = np.asarray([width / IMAGE_SIZE, height / IMAGE_SIZE] * 2, dtype=np.float32)
        return boxes * scale, scores


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def _wrapper(model):
    import torch
    from torch import nn
    from torchvision.models.detection.image_list import ImageList
    from torchvision.ops import clip_boxes_to_image

    class Export(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.head = model.backbone, model.head
            self.anchor_generator, self.box_coder = model.anchor_generator, model.box_coder
            self.register_buffer("mean", torch.tensor(model.transform.image_mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(model.transform.image_std).view(1, 3, 1, 1))

        def forward(self, images):
            features = list(self.backbone((images - self.mean) / self.std).values())
            output = self.head(features)
            image_list = ImageList(images, [(IMAGE_SIZE, IMAGE_SIZE)])
            anchors = self.anchor_generator(image_list, features)[0]
            boxes = self.box_coder.decode_single(output["bbox_regression"][0], anchors)
            boxes = clip_boxes_to_image(boxes, (IMAGE_SIZE, IMAGE_SIZE))
            scores = torch.softmax(output["cls_logits"][0], dim=-1)[:, CAT_LABEL]
            return boxes, scores
    return Export().eval()


def export(checkpoint, out, manifest, data_root=None, score_threshold=0.3):
    """Export checkpoint and validate first five development images against Torch."""
    import torch
    from experiments.detection.train import build_model, read_manifest

    checkpoint = Path(checkpoint); out = Path(out)
    if not 0 <= score_threshold <= 1:
        raise ValueError("score threshold must be in [0,1]")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(False).eval(); model.load_state_dict(state["model_state"], strict=True)
    wrapper = _wrapper(model)
    dummy = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    torch.onnx.export(wrapper, dummy, out, input_names=["images"], output_names=["boxes", "scores"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    rows = [r for r in read_manifest(Path(manifest)) if r["split"] == "development"][:5]
    root = Path(data_root) if data_root else Path("data")
    metadata = {"input_size": [1, 3, IMAGE_SIZE, IMAGE_SIZE], "input_range": "RGB float32 0..1",
                "score_threshold": score_threshold, "nms_threshold": float(model.nms_thresh),
                "topk_candidates": int(model.topk_candidates), "max_boxes": 10,
                "cat_label": CAT_LABEL, "coordinate_system": "decoded xyxy pixels in 320x320",
                "preprocess": "EXIF transpose, RGB, PIL bilinear resize, /255; normalization inside graph",
                "normalization": {"mean": model.transform.image_mean, "std": model.transform.image_std},
                "postprocess": "class 17 score >= metadata score_threshold, top-k, single-class NumPy NMS, max 10; boxes map from 320 to original pixels",
                "training_dataset": "CatFLW publisher boxes, possibly partial; training frames NOT exhaustively reviewed",
                "research_only": True, "source_data_license": "CC BY-NC 4.0; no commercial clearance assessed",
                "training": {key: state["metadata"][key] for key in ("manifest_sha256", "source_weights", "best_epoch", "epochs")},
                "model_sha256": _sha256(out),
                "checkpoint_sha256": _sha256(checkpoint), "torchvision": __import__("torchvision").__version__}
    # A small parity check catches accidental preprocessing/decode changes without
    # making the runtime depend on Torch.
    validation = {"samples": len(rows), "matched": 0, "max_abs_box_error": None, "max_abs_score_error": None}
    try:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"], sess_options=options)
        input_name = session.get_inputs()[0].name
        box_errors, score_errors = [], []
        for row in rows:
            path = Path(row["path"])
            if not path.is_absolute(): path = root / path
            image_tensor, _ = _preprocess(path)
            with torch.no_grad():
                official = model([torch.from_numpy(image_tensor[0])])[0]
            mask = official["labels"].numpy() == CAT_LABEL
            torch_boxes = official["boxes"].numpy()[mask]
            torch_scores = official["scores"].numpy()[mask]
            selected = torch_scores >= metadata["score_threshold"]
            torch_boxes, torch_scores = torch_boxes[selected][:metadata["max_boxes"]], torch_scores[selected][:metadata["max_boxes"]]
            ort_boxes, ort_scores = session.run(None, {input_name: image_tensor})
            ort_boxes, ort_scores = _select(ort_boxes, ort_scores, metadata)
            if len(torch_boxes) != len(ort_boxes):
                raise RuntimeError(f"Torch/ORT detection count mismatch for {row['sample_id']}: {len(torch_boxes)} != {len(ort_boxes)}")
            validation["matched"] += len(torch_boxes)
            if len(torch_boxes):
                box_errors.append(float(np.max(np.abs(torch_boxes - ort_boxes))))
                score_errors.append(float(np.max(np.abs(torch_scores - ort_scores))))
        if box_errors: validation["max_abs_box_error"] = max(box_errors)
        if score_errors: validation["max_abs_score_error"] = max(score_errors)
        if validation["max_abs_box_error"] is not None and validation["max_abs_box_error"] > 0.02:
            raise RuntimeError(f"Torch/ORT box error too large: {validation['max_abs_box_error']}")
        if validation["max_abs_score_error"] is not None and validation["max_abs_score_error"] > 1e-4:
            raise RuntimeError(f"Torch/ORT score error too large: {validation['max_abs_score_error']}")
    except ImportError:
        raise RuntimeError("onnxruntime is required for export parity validation")
    metadata["validation"] = validation
    sidecar = out.with_suffix(out.suffix + ".json"); sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def predict(model, manifest, split, out, threads=2, data_root="data"):
    """Run ORT and NumPy NMS, writing JSONL predictions and a JSON sidecar."""
    import onnxruntime as ort
    rows = []
    with Path(manifest).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not split:
        raise ValueError("split must be non-empty")
    rows = [r for r in rows if r.get("split") == split]
    if not rows:
        raise ValueError(f"manifest has no rows for split: {split}")
    detector = Detector(model, threads)
    metadata = detector.metadata
    result = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute(): path = Path(data_root) / path
        decode_started = time.perf_counter()
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        started = time.perf_counter()
        decode_ms = (started - decode_started) * 1000
        boxes, scores = detector.detect(image)
        elapsed = (time.perf_counter() - started) * 1000
        result.append({"sample_id": row["sample_id"], "boxes": boxes.tolist(), "scores": scores.tolist(), "detect_ms": elapsed, "decode_ms": decode_ms})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(json.dumps(r) for r in result) + ("\n" if result else ""), encoding="utf-8")
    Path(str(out) + ".json").write_text(json.dumps({"model_sha256": _sha256(model), "split": split,
                "threads": threads, "onnxruntime": ort.__version__, "manifest_sha256": _sha256(manifest),
                "timing_scope": "decode_ms: file decode + EXIF/RGB; detect_ms: PIL resize + ORT + NumPy NMS + coordinate mapping",
                "metadata": metadata}, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export"); e.add_argument("--checkpoint", required=True); e.add_argument("--out", required=True); e.add_argument("--manifest", required=True); e.add_argument("--data-root")
    e.add_argument("--score-threshold", type=float, default=0.3)
    q = sub.add_parser("predict"); q.add_argument("--model", required=True); q.add_argument("--manifest", required=True); q.add_argument("--split", default="development"); q.add_argument("--out", required=True); q.add_argument("--threads", type=int, default=2); q.add_argument("--data-root", default="data")
    a = p.parse_args(argv)
    return export(a.checkpoint, a.out, a.manifest, a.data_root, a.score_threshold) if a.command == "export" else predict(a.model, a.manifest, a.split, a.out, a.threads, a.data_root)

if __name__ == "__main__": main()
