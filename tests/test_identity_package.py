import tarfile

import pytest

from experiments.identity import package


def test_edge_archive_contains_only_runtime_and_model(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    identity = root / "experiments" / "identity"
    identity.mkdir(parents=True)
    (root / "README.md").write_text("readme", encoding="utf-8")
    (identity / "requirements-runtime.txt").write_text("Pillow\n", encoding="utf-8")
    (identity / "README.md").write_text("edge", encoding="utf-8")
    (identity / "__init__.py").write_text("", encoding="utf-8")
    (identity / "runtime.py").write_text("runtime", encoding="utf-8")
    (identity / "deploy.py").write_text("deploy", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"onnx")
    (model_dir / "model.json").write_text("{}", encoding="utf-8")
    (model_dir / "gallery.json").write_text("{}", encoding="utf-8")
    (root / ".env").write_text("SECRET=x", encoding="utf-8")
    monkeypatch.setattr(package, "ROOT", root)

    output = tmp_path / "bundle.tar.gz"
    package.build_archive(output, "edge", model_dir)
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert names == {
        "experiments/identity/requirements-runtime.txt",
        "experiments/identity/README.md",
        "experiments/identity/__init__.py",
        "experiments/identity/runtime.py",
        "experiments/identity/deploy.py",
        "model/model.onnx",
        "model/model.json",
        "model/gallery.json",
    }
    assert all(".env" not in name and "__pycache__" not in name for name in names)
    detection = root / "experiments" / "detection"
    detection.mkdir()
    for name in ("onnx.py", "containment.py", "replay.py", "README.md", "enroll.py"):
        (detection / name).write_text("research runtime", encoding="utf-8")
    (identity / "enroll.py").write_text("enroll", encoding="utf-8")
    catid = root / "src" / "catid"
    catid.mkdir(parents=True)
    for name in ("__init__.py", "contracts.py", "evaluation.py"):
        (catid / name).write_text("runtime", encoding="utf-8")
    docs = root / "docs"
    docs.mkdir()
    (docs / "detection-study-2026-09-22.md").write_text("detector study", encoding="utf-8")
    (docs / "crop-study-2026-09-22.md").write_text("crop study", encoding="utf-8")
    detector_dir = tmp_path / "detector"
    detector_dir.mkdir()
    (detector_dir / "model.onnx").write_bytes(b"detector")
    (detector_dir / "model.onnx.json").write_text("{}", encoding="utf-8")
    package.build_archive(output, "edge", model_dir, detector_dir=detector_dir)
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "detector/model.onnx.json" in names
    assert "experiments/detection/replay.py" in names
    assert "experiments/detection/containment.py" in names
    assert "experiments/detection/enroll.py" in names
    assert "experiments/identity/enroll.py" in names
    assert {"catid/__init__.py", "catid/contracts.py", "catid/evaluation.py"}.issubset(names)
    assert "docs/detection-study-2026-09-22.md" in names
    assert "docs/crop-study-2026-09-22.md" in names
    assert not any("train.py" in name for name in names)


def test_server_archive_copies_manifest_images_and_excludes_raw(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "data" / "clean" / "faces" / "cat").mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "data" / "clean.identity.jsonl").write_text(
        '{"path":"clean/faces/cat/a.png"}\n', encoding="utf-8")
    (root / "data" / "clean" / "faces" / "cat" / "a.png").write_bytes(b"png")
    (root / "data" / "raw" / "secret.jpg").write_bytes(b"secret")
    (root / ".env").write_text("TOKEN=x", encoding="utf-8")
    (root / "README.md").write_text("readme", encoding="utf-8")
    monkeypatch.setattr(package, "ROOT", root)

    output = tmp_path / "server.tar.gz"
    package.build_archive(output, "server")
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "data/clean.identity.jsonl" in names
    assert "data/clean/faces/cat/a.png" in names
    assert not any(name.startswith("data/raw/") or name == ".env" for name in names)

    with pytest.raises(ValueError):
        package.build_archive(root / "data" / "clean" / "faces" / "bad.tar.gz", "server")
