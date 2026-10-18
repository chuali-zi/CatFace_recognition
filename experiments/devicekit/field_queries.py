"""Field-only independent queries; operator truth required, no borrowed publisher labels."""
import json, sys, time
from pathlib import Path
from experiments.devicekit.enroll_field import validate_field
from experiments.detection.onnx import Detector
from experiments.detection.replay import replay
from experiments.identity.runtime import Encoder,read_image
from experiments.identity.deploy import read_gallery
from experiments.board.run import identity_metrics

def main():
    enrollment,queries,assets,model,gallery_path,out=map(Path,sys.argv[1:])
    erows=[json.loads(l) for l in enrollment.read_text().splitlines() if l.strip()]
    qrows=[json.loads(l) for l in queries.read_text().splitlines() if l.strip()]
    validate_field(qrows,query=True)
    for key in ('session_id','event_id'):
        if {r[key] for r in erows}&{r[key] for r in qrows}: raise ValueError('Enrollment/calibration/query event or session overlap')
    if {r['cat_id'] for r in erows if r['split']=='calibration_unknown'} & {r['cat_id'] for r in qrows if r['role']=='unknown'}: raise ValueError('Calibration/query stranger identity overlap')
    # Detect exact decoded-pixel leakage across all partitions (and duplicate queries).
    import hashlib
    seen=set()
    for source,rows in [(enrollment,erows),(queries,qrows)]:
        for r in rows:
            im=read_image(source.parent/r['path'])
            try: h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            finally: im.close()
            if h in seen: raise ValueError('Duplicate image pixels across field partitions')
            seen.add(h)
    config=json.loads((assets/'bundle.json').read_text()); item=config['models'][str(model)]
    detector=Detector(assets/config['detector'],1); encoder=Encoder(assets/item['model'],1); gallery=read_gallery(gallery_path,encoder)
    results=[]
    for r in qrows:
        if (r['cat_id'] in gallery['cat_ids']) != (r['role']=='known'): raise ValueError('Query identity/role mismatch')
        start=time.perf_counter()
        try:
            im=read_image(queries.parent/r['path'])
            try: prediction=replay(im,detector,encoder,gallery)
            finally: im.close()
        except Exception as exc: prediction={'status':'error','error_type':type(exc).__name__}
        prediction.update(sample_id=r['sample_id'],role=r['role'],true_cat_id=r['cat_id'],total_ms=1000*(time.perf_counter()-start))
        results.append(prediction)
    (out/'field-predictions.json').write_text(json.dumps(results,indent=2)+'\n')
    summary={'metrics':identity_metrics(results),'query_count':len(results),'independent_sessions':len({r['session_id'] for r in qrows}),'threshold':gallery['threshold'],'scope':'Operator-attested field identities; image-level diagnostic, no low-FPIR statistical guarantee; no warmup/performance claim'}
    (out/'field-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 1 if any(r['status']=='error' for r in results) else 0
if __name__=='__main__': raise SystemExit(main())
