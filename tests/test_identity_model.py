import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
np = pytest.importorskip("numpy")
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.identity.model import create_model, get_transform
from experiments.identity.runtime import preprocess
from experiments.identity.train import Rows
from experiments.identity.train import arcface_logits


@pytest.mark.parametrize("architecture,dimension", [("mobilenet_v3_small", 576), ("mobilenet_v3_large", 960)])
def test_transform_shape_and_embedding_normalization(architecture, dimension):
    image = Image.new("RGB", (7, 13), (200, 30, 90))
    x = get_transform(160)(image)
    assert x.shape == (3, 160, 160)
    model = create_model(pretrained=False, architecture=architecture)
    with torch.no_grad():
        y = model(x.unsqueeze(0))
    assert y.shape == (1, dimension)
    assert model.embedding_dim == dimension
    assert torch.allclose(y.norm(dim=1), torch.ones(1), atol=1e-5)


def test_transform_matches_runtime_preprocess_for_exif_non_square():
    image = Image.new("RGB", (7, 13), (200, 30, 90))
    # Orientation 6 is deliberately non-square and exercises EXIF transpose.
    image.getexif()[274] = 6
    expected = preprocess(image, 32)
    actual = get_transform(32)(image).numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_load_model_rejects_incomplete_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_state": {}}, path)
    from experiments.identity.model import load_model
    with pytest.raises(RuntimeError):
        load_model(path)


def test_cached_rows_match_uncached_with_same_random_state(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (9, 13), (40, 120, 200)).save(path)
    row = {"path": path.name, "cat_id": "cat"}
    uncached = Rows([row], tmp_path, {"cat": 0}, get_transform(32, augment=True), augment=True)
    cached = Rows([row], tmp_path, {"cat": 0}, get_transform(32, augment=True), cache_images=True, augment=True)
    torch.manual_seed(77); a, _ = uncached[0]
    torch.manual_seed(77); b, _ = cached[0]
    assert torch.equal(a, b)


def test_offline_crop_training_cache_and_export_input_match(tmp_path):
    from experiments.identity.runtime import Encoder, embed_rows, load_image
    path = tmp_path / "original.png"
    pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    Image.fromarray(pixels).save(path)
    row = {"sample_id": "sample", "path": path.name, "cat_id": "cat", "crop_box": [5, 4, 26, 20]}
    expected = preprocess(Image.fromarray(pixels).crop((5, 4, 26, 20)), 32)
    for cached in (False, True):
        ds = Rows([row], tmp_path, {"cat": 0}, get_transform(32), cache_images=cached)
        np.testing.assert_allclose(ds[0][0].numpy(), expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(load_image(path, 32, row["crop_box"]), expected)
    encoder = object.__new__(Encoder)
    encoder.size = 32
    encoder.encode_array = lambda array: array
    np.testing.assert_array_equal(embed_rows(encoder, [row], tmp_path)["sample"], expected)
    with pytest.raises(ValueError, match="invalid offline crop_box"):
        load_image(path, 32, [-1, 0, 20, 20])


def test_arcface_is_finite_and_reduces_target_logit():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    weights = torch.eye(2, requires_grad=True)
    labels = torch.tensor([0, 1])
    logits = arcface_logits(embeddings, weights, labels, margin=0.3, scale=30)
    cosine = torch.nn.functional.linear(embeddings, torch.nn.functional.normalize(weights, dim=1))
    assert torch.isfinite(logits).all()
    assert torch.all(logits.diag() < cosine.diag() * 30)
    logits.sum().backward()
    assert embeddings.grad is not None and weights.grad is not None
    assert torch.isfinite(embeddings.grad).all() and torch.isfinite(weights.grad).all()
    assert torch.equal(logits[0, 1], cosine[0, 1] * 30)
    opposite = torch.tensor([[-1.0, 0.0]], requires_grad=True)
    boundary = arcface_logits(opposite, weights, torch.tensor([0]))
    boundary.sum().backward()
    assert torch.isfinite(opposite.grad).all()
