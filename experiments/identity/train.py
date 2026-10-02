"""Small, reproducible MobileNetV3 identity training command."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import create_model, get_transform
from .runtime import read_image


def arcface_logits(embeddings, weights, labels, margin=0.3, scale=30.0):
    """Additive angular-margin logits with safe cosine/sine boundaries."""
    if not 0 <= margin < math.pi / 2 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("margin must be in [0, pi/2), scale finite and positive")
    # Keep angular operations in float32 even when the backbone uses AMP.
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(embeddings.float(), dim=1),
            torch.nn.functional.normalize(weights.float(), dim=1)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        target = cosine.gather(1, labels[:, None])
        sine = torch.sqrt(torch.clamp(1.0 - target.square(), min=1e-7))
        phi = target * math.cos(margin) - sine * math.sin(margin)
        phi = torch.where(target > math.cos(math.pi - margin), phi,
                          target - math.sin(math.pi - margin) * margin)
        output = cosine.clone()
        output.scatter_(1, labels[:, None], phi)
        return output * scale


class Rows(Dataset):
    def __init__(self, rows, root: Path, labels, transform, cache_images=False, augment=False):
        self.rows, self.root, self.labels, self.transform = rows, root, labels, transform
        self.augment, self.cache_images = augment, cache_images
        self.cached = None
        if cache_images:
            base_transform = get_transform(transform.size, augment=False)
            self.cached = []
            for row in rows:
                path = Path(row["path"])
                if not path.is_absolute(): path = root / path
                self.cached.append(base_transform(read_image(path, row.get("crop_box"))))

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = Path(row["path"])
        if not path.is_absolute(): path = self.root / path
        if self.cached is not None:
            x = self.cached[i].clone()
            if self.augment and torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(-1,))
        else:
            x = self.transform(read_image(path, row.get("crop_box")))
        return x, self.labels[row["cat_id"]]


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _rows(protocol, key):
    value = protocol.get(key, [])
    if isinstance(value, dict):
        value = value.get("rows", [])
    return value


def _nested_rows(value):
    if isinstance(value, dict):
        return sum((_nested_rows(v) for v in value.values()), [])
    if isinstance(value, list):
        return value if (not value or "path" in value[0]) else sum((_nested_rows(v) for v in value), [])
    return []


def _metadata(args, dev, cats, history):
    return {"model": "torchvision.mobilenet_v3_small", "pretrained": "MobileNet_V3_Small_Weights.IMAGENET1K_V1", "embedding_dim": 576, "input_size": args.size, "manifest": str(args.manifest), "protocol": str(args.protocol), "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "device": str(dev), "torch_version": torch.__version__, "training_identities": cats, "augmentation": "post-letterbox random horizontal flip (p=0.5)", "best_epoch": min(history, key=lambda x: x['validation_loss'])['epoch'] if history else None, "history": history, "diagnostic_unit": "image-level; no event-level claims", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}


def main(argv=None):
    p = argparse.ArgumentParser(description="Train MobileNetV3-Small cat identity embeddings")
    p.add_argument("--manifest", type=Path, required=True, help="identity JSONL (recorded in metadata)")
    p.add_argument("--protocol", type=Path, required=True, help="JSON with train and validation rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu"); p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=1); p.add_argument("--size", type=int, default=160)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260922); p.add_argument("--amp", action="store_true")
    p.add_argument("--cache-images", action="store_true", help="cache deterministic preprocessed tensors in RAM")
    p.add_argument("--loss", choices=("cross-entropy", "arcface"), default="cross-entropy")
    p.add_argument("--margin", type=float, default=0.3); p.add_argument("--scale", type=float, default=30.0)
    args = p.parse_args(argv)
    for name in ("epochs", "batch_size", "threads", "size"):
        if getattr(args, name) <= 0: raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.size < 32: raise ValueError("--size must be at least 32 to match export")
    if args.workers < 0: raise ValueError("--workers must be non-negative")
    if not 0 <= args.margin < math.pi / 2 or not math.isfinite(args.scale) or args.scale <= 0:
        raise ValueError("--margin must be in [0, pi/2), --scale finite and positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    torch.set_num_threads(args.threads); _seed(args.seed)
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    train_rows, val_rows = _rows(protocol, "train"), _rows(protocol, "validation")
    if not train_rows or not val_rows: raise ValueError("protocol must contain non-empty train and validation rows")
    if {str(r["path"]) for r in train_rows} & {str(r["path"]) for r in val_rows}: raise ValueError("train and validation images must be disjoint")
    cats = sorted({r["cat_id"] for r in train_rows})
    labels = {cat: i for i, cat in enumerate(cats)}
    if any(r["cat_id"] not in labels for r in val_rows): raise ValueError("validation contains unseen training identity")
    for key in ("development", "dev", "test"):
        if any(r.get("cat_id") in labels for r in _nested_rows(protocol.get(key, []))): raise ValueError(f"{key} identity overlaps training identities")
    root = args.manifest.parent
    train_ds = Rows(train_rows, root, labels, get_transform(args.size, augment=True), cache_images=args.cache_images, augment=True)
    val_ds = Rows(val_rows, root, labels, get_transform(args.size, augment=False), cache_images=args.cache_images, augment=False)
    dev = torch.device(args.device)
    model = create_model(pretrained=True, num_classes=len(cats)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(); scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    args.out.mkdir(parents=True, exist_ok=True); best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train(); total = correct = count = 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=scaler.is_enabled()):
                if args.loss == "arcface":
                    embedding = model.embedding(x)
                    logits = arcface_logits(embedding, model.head.weight, y, args.margin, args.scale)
                    accuracy_logits = arcface_logits(embedding, model.head.weight, y, 0.0, args.scale)
                else:
                    logits = model.logits(x)
                    accuracy_logits = logits
                loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * len(y); correct += (accuracy_logits.argmax(1) == y).sum().item(); count += len(y)
        model.eval(); vtotal = vcorrect = vcount = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                if args.loss == "arcface":
                    embedding = model.embedding(x)
                    logits = arcface_logits(embedding, model.head.weight, y, args.margin, args.scale)
                    accuracy_logits = arcface_logits(embedding, model.head.weight, y, 0.0, args.scale)
                else:
                    logits = model.logits(x); accuracy_logits = logits
                vtotal += criterion(logits, y).item()*len(y); vcorrect += (accuracy_logits.argmax(1)==y).sum().item(); vcount += len(y)
        rec = {"epoch": epoch, "train_loss": total/count, "train_accuracy": correct/count, "validation_loss": vtotal/vcount, "validation_accuracy": vcorrect/vcount,
               "epoch_sec": time.perf_counter() - epoch_started}
        history.append(rec); schedule.step()
        print(f"epoch {epoch}/{args.epochs} train_loss={rec['train_loss']:.4f} train_acc={rec['train_accuracy']:.4f} val_loss={rec['validation_loss']:.4f} val_acc={rec['validation_accuracy']:.4f}", flush=True)
        payload = {"model_state": model.state_dict(), "input_size": args.size, "num_classes": len(cats), "class_to_index": labels, "epoch": epoch,
                   "loss": args.loss, "margin": args.margin, "scale": args.scale,
                   "training_crop_pipeline": protocol.get("training_crop_pipeline")}
        torch.save(payload, args.out / "last.pt")
        if rec["validation_loss"] < best: best = rec["validation_loss"]; torch.save(payload, args.out / "best.pt")
        metadata = _metadata(args, dev, cats, history)
        metadata["training_crop_pipeline"] = protocol.get("training_crop_pipeline")
        metadata["elapsed_sec"] = time.perf_counter() - started
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
