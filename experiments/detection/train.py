"""Small, bounded SSDLite320 cat-face detection fine-tuning experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype


IMAGE_SIZE = 320
FACE_LABEL = 17  # COCO cat class; training semantics are explicitly cat face.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_box(box, width: int, height: int) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("boxes must contain xyxy lists")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values) or not (0 <= values[0] < values[2] <= width and 0 <= values[1] < values[3] <= height):
        raise ValueError(f"invalid xyxy box for image size {(width, height)}: {box}")
    return values


def read_manifest(path: Path) -> list[dict]:
    rows = []
    seen = set()
    seen_paths = {}
    seen_groups = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
                raise ValueError(f"line {number}: sample_id must be non-empty")
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate sample_id: {row['sample_id']}")
            seen.add(row["sample_id"])
            if not isinstance(row.get("path"), str) or not row["path"]:
                raise ValueError(f"line {number}: path must be non-empty")
            path_key = str(Path(row["path"]))
            if path_key in seen_paths:
                raise ValueError(f"duplicate path: {row['path']}")
            seen_paths[path_key] = row["split"]
            for field in ("pixel_hash", "group_id"):
                value = row.get(field)
                if value is not None:
                    key = (field, str(value))
                    if key in seen_groups and seen_groups[key] != row["split"]:
                        raise ValueError(f"{field} crosses splits: {value}")
                    seen_groups[key] = row["split"]
            if row.get("split") not in {"train", "development", "test"}:
                raise ValueError(f"line {number}: split must be train, development, or test")
            if row.get("annotation_scope") not in {"single_target", "exhaustive"}:
                raise ValueError(f"line {number}: annotation_scope must be single_target or exhaustive")
            boxes = row.get("boxes")
            if not isinstance(boxes, list) or not boxes:
                raise ValueError(f"line {number}: boxes must be a non-empty list")
            row = dict(row)
            row["boxes"] = boxes
            rows.append(row)
    if not rows:
        raise ValueError("manifest is empty")
    if not any(row["split"] == "train" for row in rows):
        raise ValueError("manifest has no train rows")
    if not any(row["split"] == "development" for row in rows):
        raise ValueError("manifest has no development rows")
    return rows


def _image(path: Path, boxes: list[list[float]], flip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        values = [_validate_box(box, width, height) for box in boxes]
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    scale_x, scale_y = IMAGE_SIZE / width, IMAGE_SIZE / height
    scaled = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in values]
    if flip:
        image = ImageOps.mirror(image)
        scaled = [[IMAGE_SIZE - x2, y1, IMAGE_SIZE - x1, y2] for x1, y1, x2, y2 in scaled]
    return convert_image_dtype(pil_to_tensor(image), torch.float32), torch.tensor(scaled, dtype=torch.float32)


class DetectionDataset(Dataset):
    def __init__(self, rows: list[dict], data_root: Path, augment: bool):
        self.rows, self.data_root, self.augment = rows, data_root, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.data_root / path
        flip = self.augment and bool(torch.rand(()) < 0.5)
        image, boxes = _image(path, row["boxes"], flip)
        target = {"boxes": boxes, "labels": torch.full((len(boxes),), FACE_LABEL, dtype=torch.int64),
                  "image_id": torch.tensor(index, dtype=torch.int64),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
                  "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])}
        return image, target, row["sample_id"]


def _collate(batch):
    images, targets, sample_ids = zip(*batch)
    return list(images), list(targets), list(sample_ids)


def build_model(pretrained: bool = True):
    if pretrained:
        return ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)


def freeze_for_finetune(model) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    # Adapt only the final MobileNet feature block in addition to the head.
    for parameter in model.backbone.features[-1].parameters():
        parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def freeze_batchnorm_stats(model) -> None:
    """Keep BN affine parameters trainable but never update running statistics."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _loss(model, loader, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train()
    freeze_batchnorm_stats(model)
    total = 0.0
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, targets, _ in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            value = sum(losses.values())
            if not torch.isfinite(value):
                raise RuntimeError("non-finite detection loss")
            if train:
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                optimizer.step()
            total += float(value.item()) * len(images)
            count += len(images)
    return total / max(count, 1)


def _save_checkpoint(path, model, metadata):
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def train(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("--out must be empty or absent")
    rows = read_manifest(args.manifest)
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "development"]
    test_rows = [row for row in rows if row["split"] == "test"]
    partial = any(row["annotation_scope"] == "single_target" for row in train_rows + dev_rows + test_rows)
    if args.threads < 1 or args.batch_size < 1 or args.epochs < 1 or args.lr <= 0:
        raise ValueError("threads, batch-size, epochs, and lr must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu here")
    device = torch.device(args.device)
    model = build_model(pretrained=True).to(device)
    parameters = freeze_for_finetune(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(DetectionDataset(train_rows, args.data_root, True), args.batch_size, shuffle=True, collate_fn=_collate)
    dev_loader = DataLoader(DetectionDataset(dev_rows, args.data_root, False), args.batch_size, shuffle=False, collate_fn=_collate)
    metadata = {"architecture": "ssdlite320_mobilenet_v3_large", "input_size": IMAGE_SIZE,
                "source_weights": "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
                "class_id": FACE_LABEL, "class_semantics": "cat face fine-tuning target; COCO pretrained class 17 means cat body",
                "annotation_warning": "single_target rows are partial target annotations; supplied boxes are positives and unlabeled faces may be penalized as background" if partial else None,
                "manifest_sha256": _sha256(args.manifest), "train_sample_ids": [r["sample_id"] for r in train_rows],
                "development_sample_ids": [r["sample_id"] for r in dev_rows], "test_sample_count": len(test_rows),
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                "seed": args.seed, "device": str(device), "threads": args.threads,
                "optimizer": "AdamW", "frozen_backbone": "all except final feature block; detection head trainable",
                "diagnostic_only": True, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "preprocessing": "EXIF transpose; RGB; PIL bilinear resize to 320x320; tensor /255; torchvision detector internal normalization",
                "postprocess": {"class_id": FACE_LABEL, "score_threshold": 0.3, "max_boxes": 10, "nms": "torchvision SSDLite model postprocess"}}
    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = _loss(model, train_loader, device, optimizer)
        dev_loss = _loss(model, dev_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, "development_loss": dev_loss,
                  "elapsed_sec": float(time.perf_counter() - epoch_started)}
        history.append(record); metadata["history"] = history; metadata["best_epoch"] = min(history, key=lambda item: item["development_loss"])["epoch"]
        _save_checkpoint(args.out / "last.pt", model, metadata)
        if dev_loss < best:
            best = dev_loss; _save_checkpoint(args.out / "best.pt", model, metadata)
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)


def predict(args):
    rows = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
    if not rows:
        raise ValueError(f"manifest has no rows for split: {args.split}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(pretrained=False); model.load_state_dict(payload["model_state"], strict=True); model.eval()
    torch.set_num_threads(args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in rows:
            path = Path(row["path"]); path = path if path.is_absolute() else args.data_root / path
            image, _ = _image(path, row["boxes"], False)
            started = time.perf_counter()
            with torch.no_grad():
                output = model([image])[0]
            elapsed = (time.perf_counter() - started) * 1000
            keep = (output["labels"] == FACE_LABEL) & (output["scores"] >= args.threshold)
            indices = torch.where(keep)[0][:args.max_boxes]
            with Image.open(path) as source:
                oriented = ImageOps.exif_transpose(source)
                original_width, original_height = oriented.size
            boxes = output["boxes"][indices].clone()
            boxes[:, [0, 2]] *= original_width / IMAGE_SIZE
            boxes[:, [1, 3]] *= original_height / IMAGE_SIZE
            result = {"sample_id": row["sample_id"], "boxes": boxes.tolist(),
                      "scores": output["scores"][indices].tolist(), "detect_ms": elapsed}
            stream.write(json.dumps(result, allow_nan=False) + "\n")
    sidecar = {"checkpoint_sha256": _sha256(args.checkpoint), "manifest_sha256": _sha256(args.manifest),
               "split": args.split, "class_id": FACE_LABEL, "score_threshold": args.threshold,
               "max_boxes": args.max_boxes, "coordinate_space": "EXIF-corrected original image xyxy",
               "preprocessing": "EXIF transpose; RGB; PIL bilinear resize to 320x320; tensor /255; torchvision detector internal normalization",
               "nms": "torchvision SSDLite model postprocess", "torch": torch.__version__, "torchvision": torchvision.__version__}
    args.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--manifest", type=Path, required=True); train_parser.add_argument("--data-root", type=Path, required=True)
    train_parser.add_argument("--out", type=Path, required=True); train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=8); train_parser.add_argument("--threads", type=int, default=4)
    train_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train_parser.add_argument("--lr", type=float, default=1e-3); train_parser.add_argument("--weight-decay", type=float, default=1e-4); train_parser.add_argument("--seed", type=int, default=20260922)
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--manifest", type=Path, required=True); predict_parser.add_argument("--data-root", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True); predict_parser.add_argument("--out", type=Path, required=True)
    predict_parser.add_argument("--split", choices=("train", "development", "test"), default="development")
    predict_parser.add_argument("--threshold", type=float, default=0.3); predict_parser.add_argument("--max-boxes", type=int, default=10); predict_parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "predict" and (args.threads < 1 or not 0 <= args.threshold <= 1 or args.max_boxes < 1):
        parser.error("invalid predict arguments")
    (train if args.command == "train" else predict)(args)


if __name__ == "__main__":
    main()
