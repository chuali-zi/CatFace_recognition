import json

import pytest

np = pytest.importorskip("numpy")

from experiments.identity.protocol import predictions_from_embeddings
from experiments.identity.runtime import make_gallery, match_gallery


def test_preprocess_preserves_pixels_and_exif_for_all_orientations():
    from PIL import Image, ImageOps
    from experiments.identity.runtime import MEAN, STD, oriented_rgb, preprocess

    pixels = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
    for mode in ("RGB", "L", "RGBA", "CMYK", "P"):
        for orientation in range(1, 9):
            source = Image.fromarray(pixels).convert(mode)
            source.getexif()[274] = orientation
            before = source.tobytes()
            # Exact reference before the copy-elision optimization.
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            side = max(rgb.size)
            square = Image.new("RGB", (side, side), (128, 128, 128))
            square.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
            resized = square.resize((32, 32), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
            expected = np.ascontiguousarray((array - MEAN) / STD)
            actual = preprocess(source, 32)
            np.testing.assert_array_equal(actual, expected)
            assert actual.flags.c_contiguous and actual.dtype == np.float32
            assert source.tobytes() == before
            assert source.getexif()[274] == orientation
            if mode == "RGB" and orientation == 1:
                assert oriented_rgb(source) is source


def test_read_image_is_decoded_after_close_and_crops_in_oriented_coordinates(tmp_path):
    from PIL import Image, ImageOps
    from experiments.identity.runtime import read_image

    pixels = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
    for suffix in ("png", "jpg", "tiff"):
        for orientation in (1, 6):
            source = Image.fromarray(pixels)
            source.getexif()[274] = orientation
            path = tmp_path / f"oriented-{orientation}.{suffix}"
            source.save(path, exif=source.getexif())
            with Image.open(path) as opened:
                expected = ImageOps.exif_transpose(opened).convert("RGB")
            actual = read_image(path)
            assert actual.size == expected.size
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            box = [1, 2, actual.width - 1, actual.height - 1]
            np.testing.assert_array_equal(np.asarray(read_image(path, box)),
                                          np.asarray(expected.crop(box)))
            # A decoded return value remains usable without its source file.
            path.unlink()
            assert actual.crop(box).tobytes() == expected.crop(box).tobytes()


def test_saved_gallery_replays_exact_evaluator_scores_and_ties():
    split = {
        "gallery": [{"cat_id": "a", "sample_id": "g1"}, {"cat_id": "b", "sample_id": "g2"}],
        "known": [{"cat_id": "a", "sample_id": "q"}],
    }
    embeddings = {"g1": np.array([1.0, 2.0], dtype=np.float32),
                  "g2": np.array([1.0, 2.0], dtype=np.float32),
                  "q": np.array([0.8, 2.1], dtype=np.float32)}
    prediction = predictions_from_embeddings(split, embeddings)[0]
    ids, vectors = make_gallery(split["gallery"], embeddings)
    gallery = json.loads(json.dumps({"cat_ids": ids, "vectors": [v.tolist() for v in vectors]}))
    cat, score = match_gallery(embeddings["q"], gallery)
    assert (cat, score) == (prediction["predicted_id"], prediction["score"])
    assert cat == "b"


def test_deploy_rejects_research_gap_threshold(tmp_path):
    from types import SimpleNamespace
    from experiments.identity.deploy import read_gallery
    from experiments.identity.runtime import model_digest

    model = tmp_path / "model.onnx"
    model.write_bytes(b"test metadata only; not an ONNX model")
    encoder = SimpleNamespace(model=model, size=160,
                              metadata={"embedding_dim": 576})
    gallery = {"model_sha256": model_digest(model), "input_size": 160,
               "aggregation": "prototype", "cat_ids": ["a"], "threshold": 0.1,
               "embedding_dim": 576, "vectors": [(np.ones(576) / 24).tolist()]}
    path = tmp_path / "gallery.json"
    path.write_text(json.dumps(gallery), encoding="utf-8")
    assert read_gallery(path, encoder) == gallery
    gallery["score_mode"] = "gap"
    path.write_text(json.dumps(gallery), encoding="utf-8")
    with pytest.raises(ValueError, match="score mode"):
        read_gallery(path, encoder)


def test_deploy_rejects_gallery_embedding_dimension_mismatch(tmp_path):
    from types import SimpleNamespace
    from experiments.identity.deploy import read_gallery
    from experiments.identity.runtime import model_digest

    model = tmp_path / "model.onnx"
    model.write_bytes(b"metadata only")
    encoder = SimpleNamespace(model=model, size=160, metadata={"embedding_dim": 3})
    gallery = {"model_sha256": model_digest(model), "input_size": 160,
               "embedding_dim": 2, "aggregation": "prototype", "cat_ids": ["a"],
               "threshold": 0.1, "vectors": [[1.0, 0.0]]}
    path = tmp_path / "gallery.json"
    path.write_text(json.dumps(gallery), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid gallery"):
        read_gallery(path, encoder)
