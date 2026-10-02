import numpy as np
import pytest

from experiments.detection.containment import suppress_contained


def test_contained_low_iou_box_is_suppressed():
    boxes = np.asarray([[0, 0, 100, 100], [10, 10, 20, 20]], dtype=float)
    np.testing.assert_array_equal(suppress_contained(boxes, [0.8, 0.7]), [0])


def test_separate_cat_boxes_are_retained():
    boxes = np.asarray([[0, 0, 20, 20], [30, 30, 50, 50]], dtype=float)
    np.testing.assert_array_equal(suppress_contained(boxes, [0.8, 0.7]), [0, 1])


def test_equal_scores_are_stable_and_suppression_is_greedy_not_transitive():
    boxes = np.asarray([[0, 0, 10, 10], [1, 0, 11, 10], [2, 0, 12, 10]], dtype=float)
    # Equal-score first box wins stable order; box 1 is removed at 0.9,
    # while box 2 is only 0.8-contained by the retained box 0.
    np.testing.assert_array_equal(suppress_contained(boxes, [0.8, 0.8, 0.7]), [0, 2])


def test_empty_and_invalid_inputs():
    np.testing.assert_array_equal(suppress_contained(np.empty((0, 4)), np.empty(0)), np.empty(0, dtype=np.int64))
    with pytest.raises(ValueError): suppress_contained([[0, 0, 1, 1]], [1, 0])
    with pytest.raises(ValueError): suppress_contained([[0, 0, 0, 1]], [1])
    with pytest.raises(ValueError): suppress_contained([[0, 0, 1, 1]], [1], 1.1)
