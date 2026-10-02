"""Fixed-model replay parity and CPU timing for preprocessing changes.

This script keeps the detector, identity model, gallery and thresholds fixed.
It records detection outputs separately from timings so a later run can prove
that a preprocessing change did not alter results.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import PIL
from PIL import ImageOps
from experiments.identity.runtime import read_image, Encoder
from experiments.identity.deploy import read_gallery
from .onnx import Detector, _sha256
from .replay import replay


ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / "experiments/runs/detection-cpu/onnx/model.onnx"
IDENTITY = ROOT / "experiments/runs/backbone-study/large224/model.onnx"
GALLERY = ROOT / "experiments/runs/crop-study/large224/fixed/detected.gallery.json"
PROTOCOL = ROOT / "experiments/runs/edge160/protocol.json"
MANIFEST = ROOT / "data/clean.identity.jsonl"


def _pct(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _rows():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    selected = [row for group in protocol["development"].values() for row in group]
    by_path = {}
    with MANIFEST.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line); by_path[row["path"]] = row
    output = []
    for row in sorted(selected, key=lambda item: item["sample_id"]):
        source = by_path.get(row["path"])
        if source is None or not source.get("original_path"):
            raise ValueError(f"missing original_path for {row['sample_id']}")
        item = dict(row); item["original_path"] = source["original_path"]; output.append(item)
    if len({row["sample_id"] for row in output}) != len(output):
        raise ValueError("development sample_ids are not unique")
    return output


def _path(relative):
    value = Path(relative)
    return value if value.is_absolute() else ROOT / "data" / value


def _environment(detector, encoder):
    try:
        import onnxruntime as ort
        ort_version = ort.__version__
    except ImportError:
        ort_version = None
    return {"platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__, "onnxruntime": ort_version,
            "pillow": PIL.__version__,
            "detector_sha256": _sha256(detector.model), "detector_metadata_sha256": detector.metadata_sha256,
            "detector_input_size": detector.metadata.get("input_size"), "identity_input_size": encoder.size,
            "identity_embedding_dim": encoder.metadata.get("embedding_dim"), "threads": 2}


def run(output, compare=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    detector = Detector(DETECTOR, threads=2)
    encoder = Encoder(IDENTITY, threads=2)
    gallery = read_gallery(GALLERY, encoder)
    cached = []
    sizes = {}
    replay_rows = []
    for row in rows:
        image = read_image(_path(row["original_path"]))
        sizes[row["sample_id"]] = [image.width, image.height]
        result = replay(image, detector, encoder, gallery)
        non_timing = {key: value for key, value in result.items()
                      if key not in {"detect_ms", "identity_and_match_ms", "frame_ms"}}
        replay_rows.append({"sample_id": row["sample_id"], "original_path": row["original_path"],
                            "image_size": sizes[row["sample_id"]], **non_timing})
        if len(cached) < 20:
            cached.append(image)
    (output / "replay.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False, allow_nan=False) for x in replay_rows) + "\n", encoding="utf-8")

    bench_images = cached
    for image in bench_images:
        replay(image, detector, encoder, gallery)
    timings = {key: [] for key in ("detect_ms", "identity_and_match_ms", "frame_ms")}
    for index in range(100):
        values = replay(bench_images[index % len(bench_images)], detector, encoder, gallery)
        for key in timings: timings[key].append(values[key])
    latency = {key: {"p50": _pct(values, 50), "p95": _pct(values, 95)} for key, values in timings.items()}
    report = {"images": len(rows), "distinct_benchmark_images": len(bench_images), "warmup": len(bench_images), "runs": 100,
              "replay_file": "replay.jsonl", "latency_ms": latency, "timing_scope":
              "decode/read_image excluded; values are returned by the actual replay() path",
              "models": {"detector": str(DETECTOR), "identity": str(IDENTITY), "gallery": str(GALLERY),
                         "identity_sha256": _sha256(IDENTITY), "gallery_sha256": _sha256(GALLERY)},
              "environment": _environment(detector, encoder), "actual_image_sizes": sizes,
              "configuration": {"sample_ids": [row["sample_id"] for row in rows], "threads": 2,
                                "detector_sha256": _sha256(DETECTOR), "identity_sha256": _sha256(IDENTITY),
                                "gallery_sha256": _sha256(GALLERY)},
              "research_only": True}
    if compare:
        previous = Path(compare)
        old_replay = previous / "replay.jsonl"
        if not old_replay.exists(): raise ValueError(f"missing comparison replay: {old_replay}")
        old = [json.loads(line) for line in old_replay.read_text(encoding="utf-8").splitlines() if line.strip()]
        if old != replay_rows:
            raise RuntimeError("non-timing replay outputs differ from --compare")
        old_report = json.loads((previous / "summary.json").read_text(encoding="utf-8"))
        if old_report.get("images") != len(rows) or old_report.get("warmup") != 20 or old_report.get("runs") != 100:
            raise RuntimeError("--compare settings differ (images/warmup/runs)")
        if old_report.get("configuration") != report["configuration"]:
            raise RuntimeError("--compare model/gallery/sample configuration differs")
        if old_report["environment"]["detector_metadata_sha256"] != detector.metadata_sha256:
            raise RuntimeError("--compare detector settings differ")
        report["compare"] = {"directory": str(previous), "non_timing_equal": True,
                              "previous_latency_ms": old_report.get("latency_ms")}
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--reference-copies", action="store_true",
                        help="experimental control: restore the former mandatory PIL copies")
    args = parser.parse_args(argv)
    if args.reference_copies:
        # Only this experiment process uses the old copy behavior. No model,
        # crop policy or recognition rule is changed by this control.
        from experiments.identity import runtime
        from . import replay as replay_module
        def copy_reference(image):
            return ImageOps.exif_transpose(image).convert("RGB")
        runtime.oriented_rgb = replay_module.oriented_rgb = copy_reference
    report = run(args.output, args.compare)
    print(json.dumps({"images": report["images"], "latency_ms": report["latency_ms"], "compare": report.get("compare")}, indent=2))


if __name__ == "__main__":
    main()
