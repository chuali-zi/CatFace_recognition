import pytest

np = pytest.importorskip("numpy")
from PIL import Image
from experiments.detection.onnx import single_crop


def test_no_fallback_or_ground_truth_selection_and_exact_rounding():
    image = Image.new("RGB", (100, 80))
    assert single_crop(image, [])[0] == "no_cat"
    assert single_crop(image, [[1, 1, 10, 10], [20, 20, 40, 40]])[0] == "unusable"
    status, box, crop = single_crop(image, [[10.2, 20.6, 90.3, 79.7]])
    assert status == "ok" and box == [10, 21, 90, 80] and crop.size == (80, 59)
    with pytest.raises(ValueError, match="invalid detector crop"):
        single_crop(image, [[-1, 0, 50, 60]])


def test_face_core_is_fixed_geometry_and_keeps_single_target_rule():
    image = Image.new("RGB", (300, 300))
    status, box, crop = single_crop(image, [[30, 30, 270, 270]], "face_core")
    assert status == "ok" and box == [70, 94, 230, 254] and crop.size == (160, 160)
    assert single_crop(image, [], "face_core")[0] == "no_cat"
    with pytest.raises(ValueError, match="unknown crop mode"):
        single_crop(image, [], "adaptive")
