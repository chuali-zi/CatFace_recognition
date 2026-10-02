"""Export and evaluate one lightweight identity candidate on a fixed protocol."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from .runtime import Encoder, embed_rows, load_image, make_gallery, model_digest


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def export_model(args):
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision
    from .model import BACKBONES, load_model

    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, device="cpu", architecture=args.architecture)
    model.eval()
    size = args.size
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    trained_ids = []
    objective = "imagenet_pretrained_features"
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if int(checkpoint["input_size"]) != size:
            raise ValueError("--size must match the checkpoint input_size")
        trained_ids = sorted(checkpoint["class_to_index"])
        objective = checkpoint.get("loss", "cross-entropy")
        if set(trained_ids) != {row["cat_id"] for row in protocol["train"]}:
            raise ValueError("Checkpoint training identities do not match this protocol")
    examples = protocol["train"][:5]
    if not examples:
        raise ValueError("Need real training images for export parity verification")
    inputs = [load_image(args.manifest.parent / row["path"], size, row.get("crop_box"))[None] for row in examples]
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(inputs[0]), str(output),
                          input_names=["image"], output_names=["embedding"],
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(output), sess_options=options,
                                    providers=["CPUExecutionProvider"])
    errors, cosines = [], []
    for array in inputs:
        with torch.inference_mode():
            expected = model(torch.from_numpy(array)).numpy()
        actual = session.run(None, {"image": array})[0]
        errors.append(float(np.max(np.abs(expected - actual))))
        cosines.append(float(np.sum(expected * actual) /
                             (np.linalg.norm(expected) * np.linalg.norm(actual))))
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    metadata = {
        "architecture": f"torchvision.{args.architecture}.features+avgpool+l2",
        "input_size": size, "embedding_dim": model.embedding_dim,
        "parameters": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "input": "float32 NCHW, batch=1, RGB",
        "preprocessing": "EXIF transpose; RGB; square pad RGB 128 centered with floor offsets; PIL bilinear resize; /255; ImageNet mean/std",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "weights_source": BACKBONES[args.architecture][1].url,
        "pretrained_weights": f"{type(BACKBONES[args.architecture][1]).__name__}.IMAGENET1K_V1",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "training_identities": trained_ids,
        "training_objective": objective,
        "precision": "fp32", "opset": 17,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "onnxruntime": ort.__version__, "python": platform.python_version(),
        "parity": {"n_real_training_images": len(inputs), "max_absolute_error": max(errors),
                   "min_cosine": min(cosines)},
        "scope": "Reviewed face crops only; no detector or image quality rejection",
    }
    if args.checkpoint and "teacher" in checkpoint:
        metadata["teacher"] = checkpoint["teacher"]
    if args.checkpoint and checkpoint.get("training_crop_pipeline"):
        metadata["training_crop_pipeline"] = checkpoint["training_crop_pipeline"]
        metadata["scope"] = "Identity embedding from externally cropped images; training crop policy is recorded separately; no integrated detector"
    write_json(output.with_suffix(".json"), metadata)
    print(json.dumps({"model": str(output), "size_mib": output.stat().st_size / 2**20,
                      "parity": metadata["parity"]}, indent=2))


def evaluate_model(args):
    from .protocol import evaluate_embeddings

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    split = protocol[args.partition]
    encoder = Encoder(args.model, args.threads)
    rows = [row for group in ("gallery", "known", "calibration_unknown", "unknown")
            for row in split[group]]
    if set(encoder.metadata.get("training_identities", [])) & {row["cat_id"] for row in rows}:
        raise ValueError("Model training identities overlap evaluation identities")
    embeddings = embed_rows(encoder, rows, args.manifest.parent)
    predictions, metrics = evaluate_embeddings(protocol, embeddings, partition=args.partition,
                                               target_fpir=args.target_fpir,
                                               aggregation=args.aggregation)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metrics.update({"partition": args.partition, "diagnostic_only": True,
                    "model": str(args.model), "protocol": str(args.protocol),
                    "aggregation": args.aggregation, "input_size": encoder.size,
                    "embedding_dim": int(encoder.metadata["embedding_dim"]),
                    "architecture": encoder.metadata.get("architecture"),
                    "n_gallery_images": len(split["gallery"]),
                    "n_gallery_identities": len({r["cat_id"] for r in split["gallery"]}),
                    "limitations": "Image-level public-data diagnostic; no verified capture sessions or detector; no event-level error claim."})
    write_json(args.out / "metrics.json", metrics)
    # Save the exact evaluated gallery for image replay on the same model.
    if args.aggregation == "prototype":
        identities, prototypes = make_gallery(split["gallery"], embeddings)
        write_json(args.out / "gallery.json", {
            "model_sha256": model_digest(args.model), "input_size": encoder.size,
            "embedding_dim": int(encoder.metadata["embedding_dim"]),
            "aggregation": "prototype", "cat_ids": identities,
            "vectors": [vector.tolist() for vector in prototypes],
            "threshold": metrics["threshold"], "diagnostic_only": True,
            "note": "This threshold is calibrated for this exact research gallery only. Recalibrate after changing enrollment or model.",
        })
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", type=Path)
    export.add_argument("--architecture", choices=["mobilenet_v3_small", "mobilenet_v3_large"],
                        default="mobilenet_v3_small", help="Large is an unfinetuned feature reference")
    export.add_argument("--size", type=int, default=160)
    export.add_argument("--out", required=True, type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--partition", choices=["development", "test"], default="development")
    evaluate.add_argument("--aggregation", choices=["prototype"], default="prototype")
    evaluate.add_argument("--target-fpir", type=float, default=0.01)
    evaluate.add_argument("--out", required=True, type=Path)
    for child in (export, evaluate):
        child.add_argument("--manifest", type=Path, default=Path("data/clean.identity.jsonl"))
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or (args.command == "export" and args.size < 32):
        parser.error("threads must be positive and size must be at least 32")
    (export_model if args.command == "export" else evaluate_model)(args)


if __name__ == "__main__":
    main()
