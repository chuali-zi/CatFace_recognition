import pytest
torch = pytest.importorskip("torch")

from experiments.identity.distill import CachedPairs, _check_protocol, similarity_loss


def test_similarity_loss_handles_different_embedding_dimensions_and_gradients():
    student = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    teacher = torch.tensor([[0., 0., 1.], [0., 1., 0.]])
    loss = similarity_loss(student, teacher)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_similarity_loss_stops_teacher_gradient_and_singleton_is_finite():
    student = torch.randn(2, 4, requires_grad=True)
    teacher = torch.randn(2, 6, requires_grad=True)
    similarity_loss(student, teacher).backward()
    assert teacher.grad is None
    one = similarity_loss(torch.ones(1, 4, requires_grad=True), torch.ones(1, 6))
    assert torch.isfinite(one) and one.item() == 0.0


def test_teacher_ids_must_not_overlap_holdout():
    protocol = {"train": [{"path": "a", "cat_id": "a"}, {"path": "aa", "cat_id": "a"}],
                "validation": [{"path": "b", "cat_id": "a"}, {"path": "bb", "cat_id": "a"}],
                "development": {"known": [{"path": "c", "cat_id": "teacher-cat"}]}}
    try:
        _check_protocol(protocol, {"training_identities": ["teacher-cat"]})
    except ValueError as exc:
        assert "development" in str(exc)
    else:
        raise AssertionError("expected teacher/holdout overlap rejection")
    protocol["development"]["known"][0]["cat_id"] = "a"
    with pytest.raises(ValueError, match="student training"):
        _check_protocol(protocol, {})


def test_cached_pairs_select_matching_flip_target(monkeypatch):
    image = torch.arange(12).reshape(1, 3, 2, 2)
    targets = torch.tensor([[[1., 0.], [0., 1.]]])
    dataset = CachedPairs(image, targets, True)
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(0.1))
    x, t = dataset[0]
    assert torch.equal(x, image[0].flip(-1))
    assert torch.equal(t, targets[0, 1])
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(0.9))
    x, t = dataset[0]
    assert torch.equal(x, image[0]) and torch.equal(t, targets[0, 0])
    dataset.augment = False
    monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(0.1))
    x, t = dataset[0]
    assert torch.equal(x, image[0]) and torch.equal(t, targets[0, 0])
