"""Research-only full-frame replay: SSDLite face boxes then frozen identity ONNX."""
import argparse
import json
import time
from pathlib import Path

from .onnx import Detector, single_crop, crop_policy
from experiments.identity.deploy import read_gallery
from experiments.identity.runtime import Encoder, match_gallery, oriented_rgb, preprocess, read_image


def calibration_scope(gallery, detector):
    pipeline = gallery.get("crop_pipeline")
    if not pipeline or pipeline.get("mode") == "legacy":
        return "New detector crops with old crop-calibrated identity gallery: NOT end-to-end calibrated"
    if pipeline.get("mode") not in {"mixed", "detected"}:
        raise ValueError("unknown gallery crop pipeline")
    crop_policy(pipeline.get("crop_mode", "head"))
    if (pipeline.get("detector_model_sha256") != detector.model_sha256 or
            pipeline.get("detector_metadata_sha256") != detector.metadata_sha256):
        raise ValueError("gallery calibration uses a different detector or detector configuration")
    return "Research gallery calibrated on detected crops for this frozen pipeline; NOT field/event validated"


def replay(image, detector, encoder, gallery):
    started = time.perf_counter()
    scope = calibration_scope(gallery, detector)
    image = oriented_rgb(image)
    boxes, scores = detector.detect(image)
    crop_mode = gallery.get("crop_pipeline", {}).get("crop_mode", "head")
    status, box, crop = single_crop(image, boxes, crop_mode)
    detected = time.perf_counter()
    result = {"status": "no_cat", "diagnostic_only": True,
              "scope": scope,
              "face_boxes": boxes.tolist(), "face_scores": scores.tolist()}
    if status == "unusable":
        result.update(status="unusable", reason="multiple cats/candidates")
    elif status == "ok":
        embedding = encoder.encode_array(preprocess(crop, encoder.size))
        cat_id, score = match_gallery(embedding, gallery)
        accepted = score >= gallery["threshold"]
        result.update(status="known" if accepted else "unknown", cat_id=cat_id if accepted else None,
                      top_candidate_id=cat_id, score=score, crop_box=box)
    result["crop_mode"] = crop_mode
    ended = time.perf_counter()
    result.update(detect_ms=(detected - started) * 1000,
                  identity_and_match_ms=(ended - detected) * 1000, frame_ms=(ended - started) * 1000)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--identity-model", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    loaded = time.perf_counter()
    detector = Detector(args.detector_model, args.threads)
    encoder = Encoder(args.identity_model, args.threads)
    gallery = read_gallery(args.gallery, encoder)
    model_load_ms = (time.perf_counter() - loaded) * 1000
    started = time.perf_counter()
    image = read_image(args.image)
    decode_ms = (time.perf_counter() - started) * 1000
    result = replay(image, detector, encoder, gallery)
    result.update(image=str(args.image), model_load_ms=model_load_ms, decode_ms=decode_ms,
                  threads=args.threads)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
