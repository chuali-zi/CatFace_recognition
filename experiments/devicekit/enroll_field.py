"""Require operator-verified identities/sessions; preserve upstream calibration evidence locally."""
import json,sys
from pathlib import Path

def validate_field(rows, query=False):
    if not rows: raise ValueError('Empty field manifest')
    if len({r.get('sample_id') for r in rows}) != len(rows): raise ValueError('Duplicate sample IDs')
    for r in rows:
        if r.get('identity_verified') is not True or not r.get('session_id') or not r.get('event_id'): raise ValueError('Verified identity, session_id and event_id required')
        if not str(r.get('sample_id','')).startswith('sample-'): raise ValueError('Use anonymous sample-NNN IDs')
        if query and r.get('role') not in ('known','unknown'): raise ValueError('Queries need known/unknown role')
    if query:
        if {r['role'] for r in rows}!={'known','unknown'}: raise ValueError('Both query roles required')
        return
    splits={}
    for r in rows:
        for kind in ('session_id','event_id'):
            key=(kind,r[kind]); splits.setdefault(key,set()).add(r['split'])
    if any(len(v)>1 for v in splits.values()): raise ValueError('Session/event leaked across splits')

def main():
    from experiments.detection.enroll import main as enroll
    manifest,assets,model,out=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4])
    rows=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    validate_field(rows)
    from experiments.devicekit.build import digest
    original_manifest_sha256 = digest(manifest)
    config=json.loads((assets/'bundle.json').read_text())
    enroll(['--manifest',str(manifest),'--detector',str(assets/config['detector']),'--model',str(assets/config['models'][model]['model']),'--out',str(out),'--threads','1'])
    # Preserve actual calibration identity scores, not just a chosen threshold.
    from experiments.devicekit.validate_model import load_models, load_gallery
    from experiments.identity.runtime import read_image
    from experiments.detection.replay import replay
    from catid.evaluation import threshold_at_fpir
    detector, encoder = load_models(assets, model)
    gallery = load_gallery(out, detector, encoder)
    records = []
    for row in rows:
        if row['split'] != 'calibration_unknown':
            continue
        try:
            image = read_image(manifest.parent / row['path'])
            try:
                prediction = replay(image, detector, encoder, gallery)
            finally:
                image.close()
            result = {'sample_id': row['sample_id'], 'status': prediction['status'], 'score': prediction.get('score')}
        except Exception as exc:
            result = {'sample_id': row['sample_id'], 'status': 'error', 'score': None, 'error_type': type(exc).__name__}
        records.append(result)
    scores = [r['score'] for r in records if r['score'] is not None]
    if len(scores) != gallery['calibration_success'] or threshold_at_fpir(scores, gallery['target_calibration_fpir']) != gallery['threshold']:
        raise ValueError('Calibration score replay did not reproduce threshold')
    if digest(manifest) != original_manifest_sha256:
        raise ValueError('Enrollment manifest changed during execution')
    evidence = Path(str(out) + '.calibration.json')
    evidence.write_text(json.dumps({'scores': records, 'threshold': gallery['threshold'],
        'target_calibration_fpir': gallery['target_calibration_fpir'],
        'acceptance_rule': 'score >= threshold', 'planned': gallery['calibration_plan'],
        'successful': gallery['calibration_success'], 'scope': 'Calibration only; not independent field FPIR'}, indent=2) + '\n')
    receipt = {'gallery_sha256': digest(out), 'model_sha256': digest(assets / config['models'][model]['model']),
        'identity_metadata_sha256': digest((assets / config['models'][model]['model']).with_suffix('.json')),
        'detector_model_sha256': detector.model_sha256, 'detector_metadata_sha256': detector.metadata_sha256,
        'enrollment_manifest_sha256': original_manifest_sha256, 'identity_truth': 'operator_attested',
        'calibration_evidence_file': evidence.name, 'calibration_evidence_sha256': digest(evidence)}
    Path(str(out) + '.provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')

if __name__=='__main__': main()
