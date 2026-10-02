import pytest

from catid import RecognitionResult


def test_known_and_unknown_results():
    assert RecognitionResult("known", "cat-a", 0.92).cat_id == "cat-a"
    assert RecognitionResult("unknown", score=0.4).cat_id is None
    assert RecognitionResult("no_cat").score is None


def test_uncertain_results_cannot_carry_identity_and_success_needs_score():
    with pytest.raises(ValueError):
        RecognitionResult("unknown", cat_id="cat-a")
    with pytest.raises(ValueError):
        RecognitionResult("known", "cat-a")
    with pytest.raises(ValueError):
        RecognitionResult("unusable")
