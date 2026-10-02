"""Build a small, relocatable server or edge archive for an identity run."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_SUFFIXES = {".py", ".toml", ".txt", ".md"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "runs"}


def _safe_add(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    """Add one path while refusing files outside the repository/model directory."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"refusing non-regular file: {source}")
    tar.add(source, arcname=arcname, recursive=False)


def _project_files(mode: str) -> list[tuple[Path, str]]:
    if mode == "edge":
        names = ["__init__.py", "runtime.py", "deploy.py"]
        files = [(ROOT / "experiments" / "identity" / name,
                  f"experiments/identity/{name}") for name in names]
        files.extend((ROOT / "experiments" / "identity" / name,
                      f"experiments/identity/{name}")
                     for name in ("requirements-runtime.txt", "README.md"))
        return files

    result: list[tuple[Path, str]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        # Keep source/docs/tests/scripts and packaging metadata, but never raw data.
        if relative.parts[0] in {"data", "artifacts"}:
            continue
        if path.suffix.lower() not in CODE_SUFFIXES and relative.as_posix() != ".gitignore":
            continue
        result.append((path, relative.as_posix()))
    return result


def _clean_rows() -> list[dict]:
    manifest = ROOT / "data" / "clean.identity.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"missing identity manifest: {manifest}")
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def _data_files() -> list[tuple[Path, str]]:
    rows = _clean_rows()
    result = [(ROOT / "data" / "clean.identity.jsonl", "data/clean.identity.jsonl")]
    if (ROOT / "data" / "README.md").is_file():
        result.append((ROOT / "data" / "README.md", "data/README.md"))
    seen: set[str] = set()
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"manifest path is not relative: {relative}")
        key = relative.as_posix()
        if key in seen:
            continue
        seen.add(key)
        candidate = ROOT / "data" / relative
        if candidate.is_symlink():
            raise ValueError(f"manifest image must not be a symlink: {relative}")
        source = candidate.resolve()
        data_root = (ROOT / "data").resolve()
        if data_root not in source.parents or not source.is_file():
            raise FileNotFoundError(f"manifest image missing: {source}")
        result.append((source, (Path("data") / relative).as_posix()))
    return result


def _model_files(model_dir: Path | None, gallery: Path | None = None,
                 required: bool = False) -> list[tuple[Path, str]]:
    if model_dir is None:
        if gallery is not None:
            raise ValueError("--gallery requires --model-dir")
        if required:
            raise ValueError("edge mode requires --model-dir")
        return []
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)
    required = ["model.onnx", "model.json"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("model-dir missing: " + ", ".join(missing))
    result = [(model_dir / name, f"model/{name}") for name in required]
    gallery = gallery or (model_dir / "gallery.json" if (model_dir / "gallery.json").is_file() else None)
    if gallery is not None:
        if gallery.is_symlink():
            raise ValueError(f"gallery file must not be a symlink: {gallery}")
        gallery = gallery.resolve()
        if not gallery.is_file():
            raise FileNotFoundError(f"gallery file missing: {gallery}")
        result.append((gallery, "model/gallery.json"))
    return result


def build_archive(out: Path, mode: str = "server", model_dir: Path | None = None,
                  gallery: Path | None = None, pretrained_weight: Path | None = None,
                  detector_dir: Path | None = None) -> int:
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    files = _project_files(mode)
    if mode == "server":
        files.extend(_data_files())
    files.extend(_model_files(model_dir, gallery, required=(mode == "edge")))
    if detector_dir is not None:
        if mode != "edge":
            raise ValueError("--detector-dir is only for the research edge replay bundle")
        for name in ("model.onnx", "model.onnx.json"):
            files.append((Path(detector_dir) / name, f"detector/{name}"))
        # Keep this bundle deliberately small: it is an offline research
        # package, not a copy of the whole source tree.  The enrollment
        # scripts use the package-local ``catid`` runtime below.
        for name in ("onnx.py", "containment.py", "replay.py", "README.md", "enroll.py"):
            files.append((ROOT / "experiments" / "detection" / name,
                          f"experiments/detection/{name}"))
        files.append((ROOT / "experiments" / "identity" / "enroll.py",
                      "experiments/identity/enroll.py"))
        for name in ("__init__.py", "contracts.py", "evaluation.py"):
            files.append((ROOT / "src" / "catid" / name, f"catid/{name}"))
        for name in ("detection-study-2026-09-22.md", "crop-study-2026-09-22.md",
                     "preprocess-optimization-2026-09-26.md", "containment-study-2026-09-26.md"):
            report = ROOT / "docs" / name
            if report.is_file():
                files.append((report, f"docs/{name}"))
    if pretrained_weight is not None:
        if pretrained_weight.is_symlink():
            raise ValueError(f"pretrained weight must not be a symlink: {pretrained_weight}")
        pretrained_weight = pretrained_weight.resolve()
        if not pretrained_weight.is_file():
            raise FileNotFoundError(f"pretrained weight missing: {pretrained_weight}")
        files.append((pretrained_weight, "weights/mobilenet_v3_small-047dcff4.pth"))
    image_root = (ROOT / "data" / "clean" / "faces").resolve()
    if image_root == out or image_root in out.parents:
        raise ValueError("--out cannot be inside the packaged clean face images")
    # Do not permit a supplied output path to be packed into itself.
    files = [(source, arcname) for source, arcname in files if source.resolve() != out]
    with tarfile.open(out, "w:gz", compresslevel=1) as tar:
        for source, arcname in files:
            _safe_add(tar, source, arcname)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output .tar.gz path")
    parser.add_argument("--mode", choices=("server", "edge"), default="server")
    parser.add_argument("--model-dir", type=Path, help="directory containing model.onnx and model.json")
    parser.add_argument("--gallery", type=Path, help="gallery.json to archive as model/gallery.json")
    parser.add_argument("--pretrained-weight", type=Path, help="optional server-side MobileNetV3 checkpoint")
    parser.add_argument("--detector-dir", type=Path, help="optional research SSDLite model and metadata for edge replay")
    args = parser.parse_args(argv)
    count = build_archive(args.out, args.mode, args.model_dir, args.gallery, args.pretrained_weight, args.detector_dir)
    print(f"wrote {args.out} ({count} files, mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
