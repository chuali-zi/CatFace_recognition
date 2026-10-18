"""One real image through the existing detector/identity/gallery pipeline."""
import json
import sys
import time
from pathlib import Path
from .validate_model import load_models, load_gallery
from experiments.detection.replay import replay
from experiments.identity.runtime import read_image


def main():
    root, model, gallery_path, image_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
    start = time.perf_counter()
    detector, encoder = load_models(root, model)
    gallery = load_gallery(gallery_path, detector, encoder)
    load_ms = 1000 * (time.perf_counter() - start)
    start = time.perf_counter()
    image = read_image(image_path)
    decoded = time.perf_counter()
    try: prediction = replay(image, detector, encoder, gallery)
    finally: image.close()
    total_ms = 1000 * (time.perf_counter() - start)
    (out / 'image-prediction.json').write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + '\n')
    # Identity labels/predictions remain local. A single image is never an accuracy benchmark.
    summary = {'status': prediction['status'], 'threshold': gallery['threshold'],
               'identity_branch_executed': prediction['status'] in ('known', 'unknown'),
               'samples': 1, 'warmup': 0, 'model_load_ms': load_ms,
               'decode_ms': 1000 * (decoded - start), 'total_ms': total_ms,
               'detect_ms': prediction['detect_ms'],
               'identity_and_match_ms': prediction['identity_and_match_ms'] if prediction['status'] in ('known', 'unknown') else None,
               'scope': 'One cold image observation; no P50/P95 or accuracy/FPIR claim'}
    (out / 'image-summary.json').write_text(json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__': main()
