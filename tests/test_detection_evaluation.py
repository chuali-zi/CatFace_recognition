import json
import pytest
pytest.importorskip("numpy")
from experiments.detection.evaluate import evaluate_rows, head_context, map_boxes, match_boxes


def test_matching_is_greedy_one_to_one_and_coordinate_mapping():
    assert map_boxes([[10, 20, 30, 40]], .5, .25) == [[20, 80, 60, 160]]
    matches = match_boxes([[0, 0, 10, 10], [0, 0, 9, 9]], [[0, 0, 10, 10]])
    assert len(matches) == 1 and matches[0][0] == 0


def test_exhaustive_counts_multiple_gt_and_precision():
    rows = [{"sample_id":"x", "boxes":[[0,0,10,10],[20,20,30,30]], "annotation_scope":"exhaustive"}]
    result = evaluate_rows(rows, {"x":[[0,0,10,10],[50,50,60,60]]}, {"x":(100,100)}, {"x":2.0})
    assert (result["tp"], result["fp"], result["fn"]) == (1,1,1)
    assert result["precision"] == .5 and result["recall"] == .5


def test_single_target_does_not_report_precision_or_extra_box_fp():
    rows = [{"sample_id":"x", "boxes":[[0,0,10,10]], "annotation_scope":"single_target"}]
    result = evaluate_rows(rows, {"x":[[0,0,10,10],[50,50,60,60]]}, {"x":(100,100)})
    assert result["matched"] == 1 and "precision" not in result and "fp" not in result


def test_predictions_selection_requires_every_selected_sample(tmp_path):
    from experiments.detection.evaluate import main
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"sample_id":"a","path":"a.png","split":"development","boxes":[],"annotation_scope":"single_target","width":10,"height":10})+"\n")
    predictions = tmp_path / "pred.jsonl"
    predictions.write_text("")
    with pytest.raises(ValueError, match="missing predictions"):
        main(["--manifest", str(manifest), "--predictions", str(predictions), "--split", "development", "--out", str(tmp_path / "out")])


def test_head_context_expands_shifts_and_clips():
    assert head_context([0, 0, 10, 10], 12, 12) == [0, 0, 12, 11]
