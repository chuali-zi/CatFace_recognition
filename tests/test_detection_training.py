import json

import pytest
torch = pytest.importorskip("torch")
from PIL import Image

from experiments.detection.train import DetectionDataset, _image, read_manifest


def test_manifest_and_bbox_resize_flip(tmp_path):
    image = tmp_path / "face.png"
    Image.new("RGB", (100, 50), (20, 30, 40)).save(image)
    Image.new("RGB", (100, 50), (20, 30, 40)).save(tmp_path / "dev.png")
    Image.new("RGB", (100, 50), (20, 30, 40)).save(tmp_path / "test.png")
    manifest = tmp_path / "boxes.jsonl"
    rows = [
        {"sample_id": "train-1", "path": "face.png", "boxes": [[10, 5, 40, 25]], "split": "train", "annotation_scope": "single_target"},
        {"sample_id": "dev-1", "path": "dev.png", "boxes": [[10, 5, 40, 25]], "split": "development", "annotation_scope": "exhaustive"},
        {"sample_id": "test-1", "path": "test.png", "boxes": [[10, 5, 40, 25]], "split": "test", "annotation_scope": "exhaustive"},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    parsed = read_manifest(manifest)
    assert len(parsed) == 3
    _, boxes = _image(image, [[10, 5, 40, 25]], flip=False)
    assert boxes.tolist() == [[32.0, 32.0, 128.0, 160.0]]
    _, flipped = _image(image, [[10, 5, 40, 25]], flip=True)
    assert flipped.tolist() == [[192.0, 32.0, 288.0, 160.0]]
    dataset = DetectionDataset([parsed[0]], tmp_path, augment=False)
    _, target, sample_id = dataset[0]
    assert sample_id == "train-1"
    assert target["labels"].tolist() == [17]


def test_manifest_rejects_bad_scope_or_box(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    row = {"sample_id": "x", "path": "x.png", "boxes": [[1, 2, 1, 4]], "split": "train", "annotation_scope": "single_target"}
    dev = dict(row, sample_id="d", path="d.png", split="development")
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(dev) + "\n", encoding="utf-8")
    assert read_manifest(manifest)
    image = tmp_path / "x.png"
    Image.new("RGB", (10, 10)).save(image)
    with pytest.raises(ValueError, match="invalid xyxy"):
        _image(image, row["boxes"], False)
    row["annotation_scope"] = "partial"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation_scope"):
        read_manifest(manifest)


def test_batchnorm_running_stats_are_frozen():
    from experiments.detection.train import freeze_batchnorm_stats
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(2))
    model.train()
    before = model[0].running_mean.clone()
    freeze_batchnorm_stats(model)
    with torch.no_grad():
        model(torch.ones(2, 2, 3, 3))
    assert not model[0].training
    assert model[0].weight.requires_grad
    assert torch.equal(before, model[0].running_mean)
