"""One CLI for data preparation and model-independent evaluation."""

import argparse
import json
from pathlib import Path

from .data import clean_duplicates, load_samples, save_samples, summarize
from .evaluation import evaluate


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cat identity data and evaluation tools; no inference backend selected.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Check sample paths and dataset composition")
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--out", type=Path)
    clean = commands.add_parser("clean", help="Remove exact-pixel duplicates from selected identity sources")
    clean.add_argument("manifest", type=Path)
    clean.add_argument("--sources", nargs="+", required=True)
    clean.add_argument("--out", type=Path, required=True)
    prepare_parser = commands.add_parser("prepare", help="Decode all sources and export deduplicated task subsets and review queues")
    prepare_parser.add_argument("manifest", type=Path)
    prepare_parser.add_argument("--near-distance", type=int, default=4)
    score = commands.add_parser("evaluate", help="Calibrate rejection and score open-set prediction JSONL")
    score.add_argument("predictions", type=Path)
    score.add_argument("--target-fpir", type=float, default=0.01)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from .prepare import prepare
            print(json.dumps(prepare(args.manifest, args.near_distance), indent=2, ensure_ascii=False))
            return
        if args.command == "evaluate":
            if args.out.resolve() == args.predictions.resolve():
                raise ValueError("Output must differ from input predictions")
            rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
            report = evaluate(rows, args.target_fpir)
        else:
            if args.out and args.out.resolve() == args.manifest.resolve():
                raise ValueError("Output must differ from input manifest")
            rows = load_samples(args.manifest)
            if args.command == "inspect":
                report = summarize(rows, args.manifest.parent)
            else:
                if args.out.resolve().parent != args.manifest.resolve().parent:
                    raise ValueError("Keep the cleaned manifest beside its input so relative image paths remain valid")
                report_path = args.out.with_suffix(".summary.json")
                if report_path.resolve() in {args.out.resolve(), args.manifest.resolve()}:
                    raise ValueError("Choose a .jsonl output filename")
                rows, report = clean_duplicates(rows, args.manifest.parent, set(args.sources))
                save_samples(args.out, rows)
                write_json(report_path, report)
                report = {key: value for key, value in report.items() if key != "removed"}
        if args.command != "clean" and args.out:
            write_json(args.out, report)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command == "inspect" and report["missing_files"]:
            raise SystemExit(1)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
