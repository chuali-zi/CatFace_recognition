"""Similarity-relation distillation from a frozen ONNX teacher to Small160."""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .model import create_model, get_transform
from .train import _nested_rows, _rows, _seed
from .runtime import model_digest


def similarity_loss(student, teacher):
    """MSE of off-diagonal pairwise cosine matrices (not a paper reimplementation)."""
    student = torch.nn.functional.normalize(student, dim=1)
    teacher = torch.nn.functional.normalize(teacher.detach(), dim=1)
    n = student.shape[0]
    if n < 2:
        return student.sum() * 0.0
    mask = ~torch.eye(n, dtype=torch.bool, device=student.device)
    return torch.nn.functional.mse_loss((student @ student.T)[mask], (teacher @ teacher.T)[mask])


def _check_protocol(protocol, teacher_meta):
    train, val = _rows(protocol, "train"), _rows(protocol, "validation")
    if len(train) < 2 or len(val) < 2: raise ValueError("train and validation need at least two rows")
    if {str(r["path"]) for r in train} & {str(r["path"]) for r in val}:
        raise ValueError("train and validation images must be disjoint")
    teacher_ids = set(teacher_meta.get("training_identities", []))
    student_ids = {row["cat_id"] for row in train}
    if not {row["cat_id"] for row in val} <= student_ids:
        raise ValueError("validation contains unseen training identity")
    for key in ("development", "dev", "test"):
        held_out = {r.get("cat_id") for r in _nested_rows(protocol.get(key, []))}
        if student_ids & held_out:
            raise ValueError(f"student training identities overlap {key}")
        if teacher_ids & held_out:
            raise ValueError(f"teacher training identities overlap {key}")
    return train, val


class CachedPairs(Dataset):
    def __init__(self, images, targets, augment):
        self.images, self.targets, self.augment = images, targets, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        flip = self.augment and torch.rand(()).item() < 0.5
        image = self.images[index]
        return (torch.flip(image, dims=(-1,)) if flip else image,
                self.targets[index, int(flip)])


def _teacher_outputs(session, input_name, tensors):
    out = []
    for tensor in tensors:
        value = session.run(None, {input_name: tensor.numpy()[None]})[0][0]
        if not np.isfinite(value).all() or np.linalg.norm(value) <= 0:
            raise ValueError("Teacher returned an invalid embedding")
        out.append(torch.from_numpy(np.asarray(value, dtype=np.float32)))
    return torch.stack(out)


def prepare(rows, root, teacher_session, teacher_input, teacher_size, student_size, augment):
    student, teacher = [], []
    for index, row in enumerate(rows, 1):
        path = Path(row["path"]); path = path if path.is_absolute() else root / path
        with Image.open(path) as image:
            sx = get_transform(student_size, augment=False)(image)
            tx = get_transform(teacher_size, augment=False)(image)
        teacher_inputs = [tx, torch.flip(tx, dims=(-1,))] if augment else [tx]
        student.append(sx)
        teacher.append(_teacher_outputs(teacher_session, teacher_input, teacher_inputs))
        if index % 100 == 0 or index == len(rows): print(f"prepared {index}/{len(rows)}", flush=True)
    return CachedPairs(torch.stack(student), torch.stack(teacher), augment)


def _save(path, model, args, epoch, labels, teacher_meta):
    torch.save({"model_state": model.state_dict(), "input_size": args.size, "num_classes": None,
                "class_to_index": labels, "epoch": epoch, "loss": "similarity-distillation",
                "teacher": teacher_meta}, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "manifest", "protocol", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--device", default="cpu")
    p.add_argument("--size", type=int, default=160); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()): raise ValueError("--out must be empty or absent")
    if (min(args.epochs, args.threads) <= 0 or args.batch_size < 2 or args.size < 32
            or not math.isfinite(args.lr) or not math.isfinite(args.weight_decay)
            or args.lr <= 0 or args.weight_decay < 0):
        raise ValueError("invalid training arguments")
    if args.device not in {"cpu", "cuda"} and not args.device.startswith("cuda:"):
        raise ValueError("--device must be cpu or cuda")
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    import onnxruntime as ort
    teacher_meta = json.loads(args.teacher.with_suffix(".json").read_text(encoding="utf-8"))
    teacher_meta = dict(teacher_meta, model_sha256=model_digest(args.teacher),
                       source=str(args.teacher))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")); train, val = _check_protocol(protocol, teacher_meta)
    _seed(args.seed); torch.set_num_threads(args.threads)
    so = ort.SessionOptions(); so.intra_op_num_threads = args.threads; so.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.teacher), sess_options=so, providers=["CPUExecutionProvider"])
    teacher_input = session.get_inputs()[0].name; root = args.manifest.parent
    train_ds = prepare(train, root, session, teacher_input, int(teacher_meta["input_size"]), args.size, True)
    val_ds = prepare(val, root, session, teacher_input, int(teacher_meta["input_size"]), args.size, False)
    del session
    labels = {cat: index for index, cat in enumerate(sorted({row["cat_id"] for row in train}))}
    dev = torch.device(args.device); model = create_model(pretrained=True, num_classes=None).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    validation = DataLoader(val_ds, args.batch_size, shuffle=False)
    args.out.mkdir(parents=True, exist_ok=True); history=[]; best=float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); total=0.0; count=0
        for x, t in loader:
            if len(x) < 2:
                continue
            x, t = x.to(dev), t.to(dev); opt.zero_grad(set_to_none=True)
            loss = similarity_loss(model(x), t)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite distillation loss")
            loss.backward(); opt.step(); total += float(loss.item()) * x.shape[0]; count += x.shape[0]
        model.eval()
        with torch.no_grad():
            predictions, targets = [], []
            for x, t in validation:
                predictions.append(model(x.to(dev)))
                targets.append(t.to(dev))
            val_loss = float(similarity_loss(torch.cat(predictions), torch.cat(targets)).item())
        if not math.isfinite(val_loss):
            raise ValueError("Non-finite validation loss")
        sched.step(); rec={"epoch":epoch,"train_loss":total/max(count,1),"validation_loss":val_loss,"lr":sched.get_last_lr()[0]}; history.append(rec)
        _save(args.out / "last.pt", model, args, epoch, labels, teacher_meta)
        if val_loss < best: best=val_loss; _save(args.out / "best.pt", model, args, epoch, labels, teacher_meta)
        metadata = {"history": history, "teacher": teacher_meta, "args": vars(args),
                    "training_identities": list(labels), "n_train": len(train), "n_validation": len(val),
                    "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
                    "loss": "similarity-distillation", "student": "mobilenet_v3_small",
                    "augmentation": "random horizontal flip p=0.5; matching original/flipped teacher targets",
                    "validation": "full-pair loss on training-identity validation images only"}
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str)+"\n", encoding="utf-8")
        print(f"epoch {epoch}/{args.epochs} train_loss={rec['train_loss']:.6f} validation_loss={val_loss:.6f}", flush=True)


if __name__ == "__main__": main()
