"""Opt-in ambiguity rejection on frozen scores or actual images; no calibration."""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def decide(ranked, threshold, min_margin=0.0):
    """Keep the original score gate; optionally reject a close runner-up."""
    threshold = finite(threshold, "threshold")
    min_margin = finite(min_margin, "min_margin")
    if min_margin < 0:
        raise ValueError("min_margin must be nonnegative")
    if not ranked.get("available", True):
        status = ranked.get("status")
        if status not in {"no_cat", "unusable", "error"}:
            status = "unusable"
        return dict(status=status, cat_id=None, accepted=False, baseline_accepted=False,
                    reason="preprocessing_failure", score=None, margin=None)
    predicted_id = ranked.get("predicted_id")
    if not isinstance(predicted_id, str) or not predicted_id:
        raise ValueError("A ranked identity must have a predicted_id")
    score = finite(ranked["score"], "score")
    similarity = finite(ranked.get("similarity", score), "similarity")
    runner_up = ranked.get("runner_up_similarity")
    margin = None
    if runner_up is not None:
        runner_up = finite(runner_up, "runner_up_similarity")
        margin = similarity - runner_up
        if margin < 0:
            raise ValueError("Runner-up similarity exceeds the winning similarity")
    baseline = score >= threshold
    reason = None
    if not baseline:
        reason = "below_score_threshold"
    elif min_margin > 0 and margin is None:
        reason = "insufficient_candidates"
    elif min_margin > 0 and margin < min_margin:
        reason = "ambiguous_identity"
    accepted = baseline and reason is None
    return dict(status="known" if accepted else "unknown",
                cat_id=predicted_id if accepted else None, accepted=accepted,
                baseline_accepted=baseline, reason=reason, score=score, margin=margin)


def summarize(rows, decision_key):
    groups = {}
    for split in ("calibration_unknown", "known", "unknown"):
        selected = [row for row in rows if row["split"] == split]
        counts = Counter(dict(queries=len(selected), correct_accepts=0, wrong_accepts=0,
                              false_accepts=0, not_accepted=0, preprocessing_failures=0))
        reasons = Counter()
        for row in selected:
            decision = row[decision_key]
            if decision["accepted"]:
                if split == "known":
                    counts["correct_accepts" if decision["cat_id"] == row["true_cat_id"] else "wrong_accepts"] += 1
                else:
                    counts["false_accepts"] += 1
            else:
                counts["not_accepted"] += 1
                counts["preprocessing_failures"] += decision["reason"] == "preprocessing_failure"
                reasons[decision["reason"]] += 1
        groups[split] = {**counts, "rejection_reasons": dict(reasons)}
    return groups


def compare_scores(assets, models, min_margin):
    reports = []
    for model in models:
        gallery = json.loads((assets / f"models/{model}/gallery.json").read_text(encoding="utf-8"))
        evidence = json.loads((assets / f"evidence/{model}-detected.json").read_text(encoding="utf-8"))
        threshold = gallery["threshold"]
        rows = []
        for split in ("calibration_unknown", "known", "unknown"):
            for ranked in evidence["predictions"][split]:
                if ranked["split"] != split or ranked["threshold"] != threshold:
                    raise ValueError("Evidence split or threshold does not match its frozen gallery")
                baseline = decide(ranked, threshold)
                if baseline["accepted"] != ranked["accepted"]:
                    raise ValueError("Default rule failed to reproduce the frozen acceptance result")
                rows.append(dict(sample_id=ranked["sample_id"], split=split,
                                 true_cat_id=ranked["cat_id"], predicted_id=ranked.get("predicted_id"),
                                 baseline=baseline, candidate=decide(ranked, threshold, min_margin)))
        reports.append(dict(model=model, threshold=threshold, min_margin=min_margin,
                            baseline=summarize(rows, "baseline"), candidate=summarize(rows, "candidate"),
                            changed_queries=sum(row["baseline"]["accepted"] != row["candidate"]["accepted"] for row in rows),
                            observations=rows))
    return reports


def recognize_images(assets, models, images, min_margin, threads):
    from experiments.detection.onnx import Detector, single_crop
    from experiments.identity.deploy import read_gallery
    from experiments.identity.matching import rank_embedding
    from experiments.identity.runtime import Encoder, preprocess, read_image

    detector = Detector(assets / "detector/model.onnx", threads)
    encoders = {model: Encoder(assets / f"models/{model}/model.onnx", threads) for model in models}
    galleries = {model: read_gallery(assets / f"models/{model}/gallery.json", encoder) for model, encoder in encoders.items()}
    results = []
    for index, path in enumerate(images):
        image = read_image(path)
        boxes, _ = detector.detect(image)
        status, box, crop = single_crop(image, boxes)
        result = dict(image_id=f"image-{index + 1:03d}", crop_status=status, crop_box=box, models={})
        for model, encoder in encoders.items():
            if crop is None:
                ranked = dict(available=False, status=status)
            else:
                embedding = encoder.encode_array(preprocess(crop, encoder.size))
                ranked = rank_embedding(embedding, galleries[model])
            threshold = galleries[model]["threshold"]
            result["models"][model] = dict(threshold=threshold, min_margin=min_margin,
                baseline=decide(ranked, threshold), candidate=decide(ranked, threshold, min_margin))
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["compare", "recognize"], default="compare")
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["small160", "large224"], default=["large224"])
    parser.add_argument("--min-margin", type=float, default=0.0, help="0 disables the experiment; positive values are uncalibrated research candidates")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--images", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        finite(args.min_margin, "min_margin")
        if args.min_margin < 0 or args.threads < 1:
            raise ValueError("Margin must be nonnegative and threads positive")
        if len(set(args.models)) != len(args.models):
            raise ValueError("Duplicate models")
        if (args.mode == "recognize") != bool(args.images):
            raise ValueError("Supply --images only for --mode recognize")
    except ValueError as exc:
        parser.error(str(exc))

    kit = args.kit.resolve()
    if not __package__:
        sys.path.insert(0, str(kit))
    from experiments.devicekit.resources import verify_assets, verify_manifest

    verify_manifest(kit, "MANIFEST.json")
    verify_assets(kit / "assets", "historical", args.models)
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(mode=args.mode, min_margin=args.min_margin, enabled=args.min_margin > 0,
        model_or_threshold_modified=False, calibrated=False, production_ready=False,
        python=platform.python_version(), execution_platform=platform.platform(),
        experiment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kit_manifest_sha256=hashlib.sha256((kit / "MANIFEST.json").read_bytes()).hexdigest(),
        kit_source_commit=json.loads((kit / "BUILD.json").read_text(encoding="utf-8"))["source_commit"],
        model_sha256={model: json.loads((kit / f"assets/models/{model}/gallery.json").read_text(encoding="utf-8"))["model_sha256"] for model in args.models},
        scope="Post-hoc development score diagnostic; no new independent accuracy or speed claim."
              if args.mode == "compare" else "Actual local image inference with a research-only decision gate; no calibrated performance claim.",
        notes=["The original score threshold and gallery are unchanged.",
               "This min-margin was proposed after viewing development failures; validate independently before enabling elsewhere.",
               "Repeated developer queries do not add independent samples; calibration and unknown-query counts are separate."])
    if args.mode == "compare":
        report["results"] = compare_scores(kit / "assets", args.models, args.min_margin)
        with (args.out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "rule", "split", "queries", "correct_accepts", "wrong_accepts", "false_accepts", "not_accepted", "preprocessing_failures"])
            for result in report["results"]:
                for rule in ("baseline", "candidate"):
                    for split, counts in result[rule].items():
                        writer.writerow([result["model"], rule, split] + [counts[k] for k in
                            ("queries", "correct_accepts", "wrong_accepts", "false_accepts", "not_accepted", "preprocessing_failures")])
    else:
        report["results"] = recognize_images(kit / "assets", args.models, args.images, args.min_margin, args.threads)
    destination = args.out / "report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Research report:", destination)
    print("Experimental gate:", "ENABLED (uncalibrated)" if report["enabled"] else "DISABLED (original rule)")
    if args.mode == "compare":
        for result in report["results"]:
            print(result["model"], "changed queries:", result["changed_queries"])
            print(json.dumps({rule: result[rule] for rule in ("baseline", "candidate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
