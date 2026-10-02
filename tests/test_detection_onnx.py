import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.detection import onnx


def test_module_does_not_import_torch_for_runtime():
    assert "torch" not in sys.modules or sys.modules["torch"] is not None
    assert onnx.IMAGE_SIZE == 320


def test_numpy_nms_suppresses_overlap_and_keeps_stable_order():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    np.testing.assert_array_equal(onnx._nms(boxes, scores, 0.5), [0, 2])


def test_select_applies_score_topk_then_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    selected, selected_scores = onnx._select(boxes, scores, {
        "score_threshold": 0.3, "nms_threshold": 0.5, "topk_candidates": 3, "max_boxes": 10,
    })
    np.testing.assert_allclose(selected, boxes[[0, 2]])
    np.testing.assert_allclose(selected_scores, scores[[0, 2]])


def test_containment_is_explicit_and_preserves_winner_coordinates():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 20, 20]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    metadata = {"score_threshold": 0.3, "nms_threshold": 0.55, "topk_candidates": 3, "max_boxes": 10}
    unchanged, unchanged_scores = onnx._select(boxes, scores, metadata)
    np.testing.assert_allclose(unchanged, boxes)
    assert len(unchanged_scores) == 2
    contained, contained_scores = onnx._select(boxes, scores, {**metadata, "containment_threshold": 0.9})
    np.testing.assert_allclose(contained, boxes[[0]])
    np.testing.assert_allclose(contained_scores, scores[[0]])


def test_preprocess_is_rgb_bilinear_320(tmp_path: Path):
    path = tmp_path / "image.png"
    Image.new("RGB", (640, 480), (64, 128, 255)).save(path)
    tensor, size = onnx._preprocess(path)
    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == np.float32
    assert size == (640, 480)
    np.testing.assert_allclose(tensor[0, :, 0, 0], np.asarray([64, 128, 255]) / 255, atol=1 / 255)
