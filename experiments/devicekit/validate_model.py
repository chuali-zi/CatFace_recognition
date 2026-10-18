"""Load real CPU ONNX models, validate graph interfaces, optionally bind a gallery."""
import json
import argparse
import sys
from pathlib import Path
from experiments.detection.onnx import Detector
from experiments.detection.replay import calibration_scope
from experiments.identity.runtime import Encoder
from experiments.identity.deploy import read_gallery


def load_models(root, model, threads=1):
    config = json.loads((root / 'bundle.json').read_text())
    detector = Detector(root / config['detector'], threads)
    encoder = Encoder(root / config['models'][model]['model'], threads)
    di, ei = detector.session.get_inputs(), encoder.session.get_inputs()
    do, eo = detector.session.get_outputs(), encoder.session.get_outputs()
    if len(di) != 1 or di[0].type != 'tensor(float)' or di[0].shape != [1, 3, 320, 320]:
        raise ValueError('Unsupported detector graph input')
    if len(do) != 2 or do[0].name != 'boxes' or do[1].name != 'scores' or len(do[0].shape) != 2 or do[0].shape[1] != 4 or len(do[1].shape) != 1:
        raise ValueError('Unsupported detector graph outputs')
    dim = int(encoder.metadata['embedding_dim'])
    if len(ei) != 1 or ei[0].type != 'tensor(float)' or ei[0].shape != [1, 3, encoder.size, encoder.size]:
        raise ValueError('Unsupported identity graph input')
    if len(eo) != 1 or eo[0].type != 'tensor(float)' or eo[0].shape != [1, dim]:
        raise ValueError('Unsupported identity graph output')
    return detector, encoder


def load_gallery(path, detector, encoder):
    gallery = read_gallery(path, encoder)
    if gallery.get('crop_pipeline', {}).get('mode') != 'detected':
        raise ValueError('Require uniformly detected/calibrated crops')
    calibration_scope(gallery, detector)
    return gallery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets',type=Path)
    parser.add_argument('model',choices=['small160','large224'])
    parser.add_argument('gallery',type=Path,nargs='?')
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    detector, encoder = load_models(args.assets, args.model, args.threads)
    if args.gallery:
        load_gallery(args.gallery, detector, encoder)
    print(json.dumps({'loaded':True,'inference_executed':False,'threads':args.threads,
        'gallery_bound':args.gallery is not None,
        'providers':{'detector':detector.session.get_providers(),'identity':encoder.session.get_providers()},
        'identity_input_size':encoder.size,'embedding_dim':encoder.metadata['embedding_dim']}))

if __name__ == '__main__': main()
