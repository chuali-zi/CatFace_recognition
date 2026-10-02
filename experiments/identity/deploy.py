"""Offline CPU deployment checks; inputs must already be cat face crops."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from .runtime import Encoder, load_image, match_gallery, model_digest


def save(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_gallery(path: Path, encoder: Encoder) -> dict:
    gallery = json.loads(path.read_text(encoding="utf-8"))
    if gallery["model_sha256"] != model_digest(encoder.model):
        raise ValueError("Gallery belongs to another model; rebuild and recalibrate it")
    vectors = np.asarray(gallery["vectors"], dtype=np.float64)
    embedding_dim = int(encoder.metadata["embedding_dim"])
    if (int(gallery.get("embedding_dim", -1)) != embedding_dim
            or vectors.shape != (len(gallery["cat_ids"]), embedding_dim) or len(vectors) == 0
            or not np.isfinite(vectors).all() or not np.isfinite(gallery["threshold"])
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-7)):
        raise ValueError("Invalid gallery vectors or threshold")
    if (gallery["input_size"] != encoder.size or gallery["aggregation"] != "prototype"
            or gallery.get("score_mode", "cosine") != "cosine"):
        raise ValueError("Gallery preprocessing/aggregation/score mode mismatch")
    return gallery


def quantize(args):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    metadata = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    # Quantization range calibration uses TRAIN images only, never evaluation probes.
    rows = list(protocol["train"])
    np.random.default_rng(20260922).shuffle(rows)
    rows = rows[:args.samples]
    if not rows:
        raise ValueError("Need training images for static quantization")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rows = iter(rows)

        def get_next(self):
            row = next(self.rows, None)
            if row is None:
                return None
            return {"image": load_image(args.manifest.parent / row["path"],
                                        metadata["input_size"])[None]}

    if args.out.resolve() == args.model.resolve():
        raise ValueError("Quantization output must differ from the FP32 model")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prepared = args.out.with_name(args.out.stem + ".preprocessed.onnx")
    quant_pre_process(str(args.model), str(prepared))
    quantize_static(str(prepared), str(args.out), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv", "Gemm", "MatMul"])
    metadata.update({"precision": "static-int8-qdq-s8s8", "parent_model": str(args.model),
                     "quantization_samples": [row["sample_id"] for row in rows],
                     "quantization_data": "training partition only; not threshold calibration",
                     "parity": None})
    save(args.out.with_suffix(".json"), metadata)
    encoder = Encoder(args.out, args.threads)
    reference = Encoder(args.model, args.threads)
    cosine = []
    for row in rows[:10]:
        path = args.manifest.parent / row["path"]
        cosine.append(float(np.dot(encoder.encode(path), reference.encode(path))))
    report = {"model": str(args.out), "size_mib": args.out.stat().st_size / 2**20,
              "n_training_calibration": len(rows), "min_training_cosine_to_fp32": min(cosine),
              "mean_training_cosine_to_fp32": float(np.mean(cosine)),
              "next": "Evaluate identity metrics with a fresh threshold; cosine alone is not acceptance."}
    save(args.out.with_suffix(".quantization.json"), report)
    print(json.dumps(report, indent=2))


def benchmark(args):
    import onnxruntime as ort
    import psutil

    started = time.perf_counter()
    encoder = Encoder(args.model, args.threads)
    load_ms = (time.perf_counter() - started) * 1000
    gallery = read_gallery(args.gallery, encoder) if args.gallery else None
    if args.images:
        paths = args.images
    else:
        if not args.manifest:
            raise ValueError("Supply --images or --manifest")
        rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [args.manifest.parent / row["path"] for row in rows[:20]]
    if not paths:
        raise ValueError("No images to benchmark")
    array = load_image(paths[0], encoder.size)
    first = time.perf_counter()
    encoder.encode_array(array)
    first_ms = (time.perf_counter() - first) * 1000
    for _ in range(args.warmup):
        encoder.encode_array(array)
    stages = {"decode_preprocess": [], "embedding": [], "matching": [], "face_crop_total": []}
    for i in range(args.runs):
        begin = time.perf_counter()
        array = load_image(paths[i % len(paths)], encoder.size)
        decoded = time.perf_counter()
        vector = encoder.encode_array(array)
        encoded = time.perf_counter()
        if gallery:
            match_gallery(vector, gallery)
        done = time.perf_counter()
        stages["decode_preprocess"].append((decoded - begin) * 1000)
        stages["embedding"].append((encoded - decoded) * 1000)
        stages["matching"].append((done - encoded) * 1000)
        stages["face_crop_total"].append((done - begin) * 1000)
    memory = psutil.Process().memory_info()
    peak = getattr(memory, "peak_wset", None)
    if peak is None and platform.system() != "Windows":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() != "Darwin":
            peak *= 1024
    report = {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
        "python": platform.python_version(), "onnxruntime": ort.__version__,
        "providers": encoder.session.get_providers(), "threads": args.threads,
        "model": str(args.model), "precision": encoder.metadata["precision"],
        "architecture": encoder.metadata.get("architecture"),
        "embedding_dim": int(encoder.metadata["embedding_dim"]),
        "model_mib": args.model.stat().st_size / 2**20,
        "input_size": encoder.size, "warmup": args.warmup, "runs": args.runs,
        "distinct_images": len(paths), "gallery_size": len(gallery["cat_ids"]) if gallery else 0,
        "session_load_ms": load_ms, "first_inference_ms": first_ms,
        "process_rss_mib": memory.rss / 2**20,
        "process_peak_rss_mib": peak / 2**20 if peak is not None else None,
        "latency_ms": {name: {"p50": float(np.percentile(values, 50)),
                              "p95": float(np.percentile(values, 95))}
                       for name, values in stages.items()},
        "scope": "Face crops only; excludes detector/camera. Process peak includes imports and model load. Repeated disk images benefit from OS caching. No extrapolation to another CPU.",
    }
    save(args.out, report)
    print(json.dumps(report, indent=2))


def recognize(args):
    encoder = Encoder(args.model, args.threads)
    gallery = read_gallery(args.gallery, encoder)
    cat, score = match_gallery(encoder.encode(args.image), gallery)
    accepted = score >= gallery["threshold"]
    print(json.dumps({"status": "known" if accepted else "unknown",
                      "cat_id": cat if accepted else None, "score": score,
                      "threshold": gallery["threshold"], "diagnostic_only": True,
                      "input_assumption": "pre-cropped cat face; no no_cat/unusable detection"},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    quant = sub.add_parser("quantize")
    quant.add_argument("--protocol", type=Path, required=True)
    quant.add_argument("--manifest", type=Path, default=Path("data/clean.identity.jsonl"))
    quant.add_argument("--samples", type=int, default=100)
    quant.add_argument("--out", type=Path, required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--images", type=Path, nargs="+")
    bench.add_argument("--manifest", type=Path)
    bench.add_argument("--gallery", type=Path)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--runs", type=int, default=100)
    bench.add_argument("--out", type=Path, required=True)
    rec = sub.add_parser("recognize", help="Replay an already cropped face against an evaluated gallery")
    rec.add_argument("--image", type=Path, required=True)
    rec.add_argument("--gallery", type=Path, required=True)
    for child in (quant, bench, rec):
        child.add_argument("--model", type=Path, required=True)
        child.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or (args.command == "benchmark" and (args.runs < 1 or args.warmup < 0)) or (args.command == "quantize" and args.samples < 1):
        parser.error("Invalid thread/sample/run count")
    {"quantize": quantize, "benchmark": benchmark, "recognize": recognize}[args.command](args)


if __name__ == "__main__":
    main()
