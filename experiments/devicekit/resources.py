"""Select and verify only the resources actually consumed by a requested mode."""
import json
import math
from pathlib import Path
from .build import digest

HISTORY_MODES = {'historical', 'smoke', 'benchmark', 'all'}
MODEL_MODES = HISTORY_MODES | {'model-check', 'recognize', 'enroll', 'field'}


def safe_file(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe asset path')
    result = root / p
    if result.is_symlink() or root.resolve() not in result.resolve().parents:
        raise ValueError('Asset escapes root')
    return result


def verify_manifest(root, name, required=None):
    entries = json.loads(safe_file(root, name).read_text(encoding='utf-8'))['files']
    if not entries:
        raise ValueError('Empty manifest')
    indexed = {}
    for entry in entries:
        if entry['path'] in indexed:
            raise ValueError('Duplicate manifest path')
        indexed[entry['path']] = entry
    selected = set(indexed) if required is None else set(required)
    if not selected <= indexed.keys():
        raise ValueError('Required resource not sealed')
    for name in selected:
        entry = indexed[name]
        f = safe_file(root, name)
        if not f.is_file() or f.stat().st_size != entry['bytes'] or digest(f) != entry['sha256']:
            raise ValueError('Integrity mismatch: ' + name)
    return selected


def plan(mode, models, external_gallery=False):
    """Human-readable minimum plan, available even when bundle.json is absent."""
    if mode not in MODEL_MODES:
        return []
    paths = ['ASSET_MANIFEST.json', 'bundle.json', 'provenance.json',
             'detector/model.onnx', 'detector/model.onnx.json']
    for model in models:
        paths += [f'models/{model}/model.onnx', f'models/{model}/model.json']
        if mode in HISTORY_MODES or (mode == 'recognize' and not external_gallery):
            paths += [f'models/{model}/gallery.json', f'calibration evidence for {model}']
    if mode in HISTORY_MODES:
        paths += ['queries.jsonl (149: 29 known / 120 unknown)',
                  '149 referenced original images', 'frozen protocol JSON']
    return paths


def validate_metadata(root, config, models):
    """Enforce the preprocessing that the shipped runtime actually implements."""
    dm = json.loads(safe_file(root, config['detector'] + '.json').read_text())
    if dm.get('model_sha256') != digest(safe_file(root, config['detector'])):
        raise ValueError('Detector metadata hash mismatch')
    if dm.get('input_size') != [1, 3, 320, 320] or dm.get('cat_label') != 17:
        raise ValueError('Unsupported detector contract')
    if dm.get('input_range') != 'RGB float32 0..1' or 'normalization inside graph' not in dm.get('preprocess', ''):
        raise ValueError('Detector preprocessing not attested')
    for key in ('score_threshold', 'nms_threshold'):
        value = dm[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid detector threshold')
    for model in models:
        size, dim = {'small160': (160, 576), 'large224': (224, 960)}[model]
        model_path = safe_file(root, config['models'][model]['model'])
        im = json.loads(model_path.with_suffix('.json').read_text())
        if im.get('input_size') != size or im.get('embedding_dim') != dim:
            raise ValueError('Unsupported identity contract')
        if im.get('mean') != [.485, .456, .406] or im.get('std') != [.229, .224, .225]:
            raise ValueError('Identity normalization mismatch')
        expected = 'EXIF transpose; RGB; square pad RGB 128 centered with floor offsets; PIL bilinear resize; /255; ImageNet mean/std'
        if im.get('preprocessing') != expected or im.get('precision') != 'fp32':
            raise ValueError('Identity preprocessing/precision mismatch')


def verify_assets(root, mode, models, external_gallery=False):
    required = {'bundle.json', 'provenance.json'}
    verify_manifest(root, 'ASSET_MANIFEST.json', required)
    config = json.loads(safe_file(root, 'bundle.json').read_text())
    provenance = json.loads(safe_file(root, 'provenance.json').read_text())
    # This verifier checks technical completeness, not blanket redistribution rights.
    # Preserve the source/license record; distribution_approved is informational.
    # A false public-distribution declaration must not prohibit authorized local use.
    if not provenance.get('sources') or not provenance.get('licenses'):
        raise ValueError('Model source/license records missing')
    required.update([config['detector'], config['detector'] + '.json'])
    frozen_galleries = mode in HISTORY_MODES or (mode == 'recognize' and not external_gallery)
    for model in models:
        item = config['models'][model]
        required.update([item['model'], Path(item['model']).with_suffix('.json').as_posix()])
        if frozen_galleries:
            required.add(item['gallery'])
            evidence = provenance.get('gallery_evidence', {}).get(model)
            if not evidence:
                raise ValueError('Selected gallery has no calibration evidence')
            for entry in evidence:
                required.add(entry['path'])
                if digest(safe_file(root, entry['path'])) != entry['sha256']:
                    raise ValueError('Gallery evidence hash mismatch')
    if mode in HISTORY_MODES:
        required.add(config['manifest'])
        protocol_path = config['historical_protocol']
        required.add(protocol_path)
        rows = [json.loads(l) for l in safe_file(root, config['manifest']).read_text().splitlines() if l.strip()]
        if len(rows) != 149 or sum(r['role'] == 'known' for r in rows) != 29 or sum(r['role'] == 'unknown' for r in rows) != 120:
            raise ValueError('Historical protocol requires all 149 queries (29 known / 120 unknown)')
        protocol = json.loads(safe_file(root, protocol_path).read_text())['development']
        expected = {(r['sample_id'], r['cat_id'], role) for role in ('known', 'unknown') for r in protocol[role]}
        observed = {(r['sample_id'], r['cat_id'], r['role']) for r in rows}
        if len(observed) != 149 or observed != expected:
            raise ValueError('Queries do not match frozen protocol')
        required.update(r['path'] for r in rows)
    verify_manifest(root, 'ASSET_MANIFEST.json', required)
    validate_metadata(root, config, models)
    hashes = {name: digest(safe_file(root, name)) for name in sorted(required)}
    return config, hashes


def verify_local_gallery(path, root, config, model):
    """Only locally generated galleries with a complete, hash-bound calibration receipt."""
    g = json.loads(path.read_text())
    receipt = json.loads(Path(str(path) + '.provenance.json').read_text())
    if receipt.get('gallery_sha256') != digest(path):
        raise ValueError('Local gallery receipt mismatch')
    expected = {'model_sha256': digest(safe_file(root, config['models'][model]['model'])),
                'detector_model_sha256': digest(safe_file(root, config['detector'])),
                'detector_metadata_sha256': digest(safe_file(root, config['detector'] + '.json')),
                'identity_metadata_sha256': digest(safe_file(root, str(Path(config['models'][model]['model']).with_suffix('.json'))))}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError('Local gallery model/preprocessing mismatch')
    evidence = path.parent / receipt['calibration_evidence_file']
    if evidence.parent.resolve() != path.parent.resolve() or digest(evidence) != receipt['calibration_evidence_sha256']:
        raise ValueError('Local calibration evidence missing/mismatch')
    if receipt.get('identity_truth') != 'operator_attested' or not receipt.get('enrollment_manifest_sha256'):
        raise ValueError('Local gallery truth/provenance missing')
    if g.get('crop_pipeline', {}).get('mode') != 'detected' or g.get('calibration_success', 0) < 1:
        raise ValueError('Uncalibrated local gallery')
    calibration = json.loads(evidence.read_text())
    scores = [r['score'] for r in calibration['scores'] if r['score'] is not None]
    target = calibration['target_calibration_fpir']
    if not scores or not all(math.isfinite(v) for v in scores) or not 0 <= target < 1:
        raise ValueError('Invalid calibration scores/target')
    threshold = math.nextafter(sorted(scores, reverse=True)[math.floor(target * len(scores))], math.inf)
    if threshold != g['threshold'] or threshold != calibration['threshold'] or len(scores) != g['calibration_success'] or target != g['target_calibration_fpir']:
        raise ValueError('Local threshold does not reproduce from calibration evidence')
    return {'gallery_sha256': digest(path), 'receipt_sha256': digest(Path(str(path) + '.provenance.json')),
            'calibration_evidence_sha256': digest(evidence), 'scope': 'Local calibration diagnostic; no field FPIR guarantee'}
