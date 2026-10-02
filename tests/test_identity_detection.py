import pytest

pytest.importorskip("numpy")
from PIL import Image

from experiments.identity.detect import iou, resize_for_detection


def test_resize_and_box_iou_use_original_coordinates():
    image, scale = resize_for_detection(Image.new("RGB", (1000, 500)), 640)
    assert image.size == (640, 320)
    assert scale == (0.64, 0.64)
    assert iou([100, 50, 300, 250], [100, 50, 300, 250]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
