import pytest

pytest.importorskip("numpy")

from experiments.detection.prepare import group_and_split


def test_same_pixels_and_mirrored_near_duplicates_never_cross_split():
    rows = []
    for i in range(30):
        rows.append(dict(pixel_hash=f"{i:064x}", dhash=f"{i * 7919:016x}",
                         dhash_mirror=f"{i * 65537:016x}", path=f"{i}.png",
                         bounding_box=[1, 2, 10, 12], width=30, height=20))
    rows.append(dict(rows[0], path="same-pixels.png", bounding_box=[1, 2, 10, 11]))
    rows[1]["dhash_mirror"] = rows[0]["dhash"]
    result = group_and_split(rows)
    assert len(result) == 30
    assert result == group_and_split(list(reversed(rows)))
    assert result[0]["split"] == result[1]["split"]
    assert result[0]["group_id"] == result[1]["group_id"]
    assert len(result[0]["duplicate_paths"]) == 2
    assert len(result[0]["boxes"]) == 1
    assert all(row["annotation_scope"] == "single_target" for row in result)
