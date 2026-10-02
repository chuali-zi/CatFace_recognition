"""Fixed repeated-gallery diagnostics on the original held-out identity pool."""

import argparse
import json
from pathlib import Path

import numpy as np

from catid.evaluation import evaluate
from .runtime import Encoder, embed_rows, model_digest


METHODS = ("prototype:cosine", "max:cosine", "prototype:gap", "max:gap")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cached_embeddings(model, manifest, protocol, partition, cache, threads):
    rows = [row for group in protocol[partition].values() for row in group]
    rows = sorted(rows, key=lambda row: row["sample_id"])
    encoder = Encoder(model, threads)
    if "embedding_dim" not in encoder.metadata:
        raise ValueError("model metadata must define embedding_dim for study cache validation")
    try:
        embedding_dim = int(encoder.metadata["embedding_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError("model metadata embedding_dim must be a positive integer") from exc
    if embedding_dim <= 0:
        raise ValueError("model metadata embedding_dim must be a positive integer")
    if set(encoder.metadata.get("training_identities", [])) & {row["cat_id"] for row in rows}:
        raise ValueError("Model training identities overlap the evaluation pool")
    source_state = []
    for row in rows:
        info = (manifest.parent / row["path"]).stat()
        source_state.append([row["sample_id"], row["cat_id"], row["path"], info.st_size, info.st_mtime_ns])
    metadata = {"model_sha256": model_digest(model), "manifest_sha256": model_digest(manifest),
                "model_metadata_sha256": model_digest(model.with_suffix(".json")),
                "input_size": encoder.size, "embedding_dim": embedding_dim,
                "partition": partition, "source_state": source_state}
    sidecar = cache.with_suffix(".json")
    if cache.exists() and sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8")) == metadata:
        with np.load(cache, allow_pickle=False) as data:
            ids, vectors = data["sample_ids"].tolist(), data["embeddings"]
        if ids != [row["sample_id"] for row in rows] or vectors.shape != (len(rows), embedding_dim) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding cache")
        print(f"reuse {len(rows)} embeddings: {cache}", flush=True)
        return dict(zip(ids, vectors))
    embeddings = embed_rows(encoder, rows, manifest.parent)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, sample_ids=np.asarray(list(embeddings)), embeddings=np.stack(list(embeddings.values())))
    save(sidecar, metadata)
    return embeddings


def summarize(results):
    fields = ("top1", "dir", "fpir", "known_misidentification_rate", "known_reject_rate")
    report = {"n_episodes": len(results)}
    for field in fields:
        values = [row[field] for row in results]
        report[field] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                         "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
                         "min": float(min(values)), "max": float(max(values))}
    report["note"] = "Macro average across reused images/identities; not independent trials or additional sample size."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/clean.identity.jsonl"))
    parser.add_argument("--models", nargs="+", required=True, help="name=path/to/model.onnx")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--partition", choices=["development", "test"], default="development")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--target-fpir", type=float, default=0.01)
    parser.add_argument("--paired-enrollment", action="store_true",
                        help="hold gallery identities and all probes fixed when comparing 3 vs 5 images")
    parser.add_argument("--cache-dir", type=Path, help="reuse embeddings across study designs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.threads < 1:
        parser.error("seeds and threads must be positive")
    from .episodes import build_paired_suite, build_suite
    from .matching import predictions

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    builder = build_paired_suite if args.paired_enrollment else build_suite
    suite = builder(protocol, partition=args.partition, seeds=range(args.seeds))
    design = ("Original identities fixed; 2/5 gallery cats x 3/5 enrollment images x fixed seeds. "
              "No scores used in sampling. Separate threshold per gallery/method from calibration_unknown only. ")
    if args.paired_enrollment:
        design += ("Within each gallery size/seed, identities and all probes are fixed; "
                   "3-image galleries are subsets of 5-image galleries. The 2 unused enrollment images "
                   "per cat are excluded from 3-image episodes. Gallery-size comparisons remain confounded by unknown composition.")
    else:
        design += ("Models/methods are paired within an episode; gallery identity eligibility and query sets differ "
                   "across enrollment settings, so this does not isolate the effect of adding enrollment images. "
                   "Gallery-size comparisons also change unknown composition.")
    save(args.out / "suite.json", {
        "partition": args.partition, "diagnostic_only": True, "repeated_images": True,
        "paired_enrollment": args.paired_enrollment,
        "methods_predeclared": args.methods, "target_fpir": args.target_fpir, "episodes": suite,
        "design": design,
    })
    summary = {"partition": args.partition, "unique_pool_identities": len({row["cat_id"] for group in protocol[args.partition].values() for row in group}),
               "unique_pool_images": sum(len(group) for group in protocol[args.partition].values()),
               "target_fpir": args.target_fpir, "diagnostic_only": True,
               "paired_enrollment": args.paired_enrollment, "candidates": {}}
    model_names = set()
    for spec in args.models:
        name, separator, model_path = spec.partition("=")
        if not separator or not name or name in model_names or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            parser.error("Each model needs a unique lowercase name=path")
        model_names.add(name)
        model = Path(model_path)
        embeddings = cached_embeddings(model, args.manifest, protocol, args.partition,
                                       (args.cache_dir or args.out / "cache") / f"{name}.npz", args.threads)
        for method in args.methods:
            aggregation, score_mode = method.split(":")
            key = f"{name}-{aggregation}-{score_mode}"
            destination = args.out / key
            destination.mkdir(parents=True, exist_ok=True)
            results = []
            with (destination / "episodes.jsonl").open("w", encoding="utf-8") as stream:
                for episode in suite:
                    rows = predictions(episode["split"], embeddings, aggregation, score_mode)
                    metrics = evaluate(rows, target_fpir=args.target_fpir)
                    metrics.update({k: episode[k] for k in ("episode_id", "seed", "gallery_size", "enrollment")})
                    results.append(metrics)
                    stream.write(json.dumps({"metrics": metrics, "predictions": rows}, ensure_ascii=False, allow_nan=False) + "\n")
            report = summarize(results)
            model_metadata = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
            report.update({"model": str(model), "model_sha256": model_digest(model),
                           "input_size": int(model_metadata["input_size"]),
                           "embedding_dim": int(model_metadata["embedding_dim"]),
                           "architecture": model_metadata.get("architecture"),
                           "aggregation": aggregation, "score_mode": score_mode,
                           "by_setting": {f"g{g}-e{e}": summarize([row for row in results if row["gallery_size"] == g and row["enrollment"] == e])
                                          for g in (2, 5) for e in (3, 5)}})
            save(destination / "summary.json", report)
            summary["candidates"][key] = report
            print(f"{key}: macro DIR={report['dir']['mean']:.3f}, FPIR={report['fpir']['mean']:.3f}, DIR p10={report['dir']['p10']:.3f}, FPIR max={report['fpir']['max']:.3f}", flush=True)
    save(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
