from types import SimpleNamespace

import pytest
pytest.importorskip("numpy")

from experiments.detection.replay import calibration_scope


def test_detected_gallery_requires_the_same_detector_configuration():
    detector = SimpleNamespace(model_sha256="model", metadata_sha256="threshold-07")
    gallery = {"crop_pipeline": {"mode": "detected", "detector_model_sha256": "model",
                                "detector_metadata_sha256": "threshold-07"}}
    assert "calibrated on detected crops" in calibration_scope(gallery, detector)
    gallery["crop_pipeline"]["detector_metadata_sha256"] = "threshold-03"
    with pytest.raises(ValueError, match="different detector"):
        calibration_scope(gallery, detector)
    assert "NOT end-to-end calibrated" in calibration_scope({}, detector)
