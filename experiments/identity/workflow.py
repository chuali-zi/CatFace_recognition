"""Run the fixed development experiment on a CPU or CUDA training host."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/clean.identity.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gallery-size", type=int, default=5)
    parser.add_argument("--enrollment", type=int, default=3)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--loss", choices=["cross-entropy", "arcface"], default="cross-entropy")
    args = parser.parse_args()
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install GPU PyTorch or explicitly choose --device cpu.")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Choose a new output directory to preserve prior runs")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = args.out / "protocol.json"

    def run(module, *arguments):
        command = [sys.executable, "-m", f"experiments.identity.{module}", *map(str, arguments)]
        print("\nRUN " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)

    run("protocol", "--manifest", args.manifest, "--out", protocol,
        "--gallery-size", args.gallery_size, "--enrollment", args.enrollment)
    common = ["--manifest", args.manifest, "--protocol", protocol, "--threads", args.threads]
    baseline = args.out / "baseline"
    trained = args.out / "trained"
    run("run", "export", *common, "--out", baseline, "--size", args.size)
    run("run", "evaluate", *common, "--model", baseline / "model.onnx", "--out", baseline / "development")
    run("train", *common, "--out", trained, "--device", args.device, "--epochs", args.epochs,
        "--batch-size", args.batch_size, "--workers", args.workers, "--size", args.size,
        "--loss", args.loss,
        *(["--cache-images"] if args.cache_images else []))
    run("run", "export", *common, "--checkpoint", trained / "best.pt", "--out", trained, "--size", args.size)
    run("run", "evaluate", *common, "--model", trained / "model.onnx", "--out", trained / "development")
    if args.int8:
        quantized = args.out / "trained-int8"
        run("deploy", "quantize", *common, "--model", trained / "model.onnx", "--out", quantized / "model.onnx")
        run("run", "evaluate", *common, "--model", quantized / "model.onnx", "--out", quantized / "development")
    print(f"Development results are in {args.out}. Final test remains untouched. Benchmark each model on the target board.")


if __name__ == "__main__":
    main()
