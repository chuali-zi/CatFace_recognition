"""Check all retained annotations geometrically and score their annotated face crops."""
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from clean import DATA, OUT, open_image, quality, read, save


def check(row):
    points=np.asarray(row['landmarks'],dtype=float)
    lo,hi=points.min(0),points.max(0)
    w,h=row['width'],row['height']
    flags=[]
    if len({tuple(p) for p in points})!=len(points): flags.append('overlapping_landmark_coordinates')
    if row['source']=='catface':
        eye=np.linalg.norm(points[0]-points[1])
        box=[max(0,int(lo[0]-(hi[0]-lo[0])*.1)),max(0,int(lo[1]-(hi[1]-lo[1])*.08)),
             min(w,int(hi[0]+(hi[0]-lo[0])*.1)),min(h,int(hi[1]+(hi[1]-lo[1])*.25))]
    else:
        # CatFLW center points: left eye 3, right eye 1 (8 is an eye corner).
        eye=np.linalg.norm(points[3]-points[1])
        box=[int(v) for v in row['bounding_box']]
    if eye<16: flags.append('small_interocular_distance')
    if min(box[2]-box[0],box[3]-box[1])<96: flags.append('small_annotated_face')
    # CatFLW boxes are part of the publisher annotation, not inferred identity targets.
    if not np.all((points>=[box[0],box[1]]) & (points<=[box[2],box[3]])):
        flags.append('points_outside_face_box')
    crop=open_image(row['path']).crop(box)
    q=quality(crop)
    if q['sharpness']<100: flags.append('soft_annotated_face')
    if q['contrast']<25: flags.append('low_annotated_face_contrast')
    return dict(row,annotation_face_box=box,face_quality=q,cleaning_flags=flags,quality_version=2,
                annotation_review_status='pending_semantic_review')


if __name__=='__main__':
    rows=read(DATA/'prepared.landmarks.jsonl')
    cache=OUT/'landmark-quality.jsonl'
    done={r['path']:r for r in read(cache)} if cache.exists() else {}
    def cached(row):
        previous=done.get(row['path'])
        matching=previous and previous.get('quality_version')==2 and all(
            previous.get(key)==row.get(key) for key in ('pixel_hash','landmarks','bounding_box','width','height'))
        return previous if matching else check(row)
    result=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(cached,rows),1):
            result.append(r)
            if i%500==0: print(i,len(rows),flush=True)
    pairs=defaultdict(dict)
    for r in result: pairs[r['pixel_hash']][r['source']]=r
    disagreements=[]
    for group in pairs.values():
        if 'catface' not in group or 'catflw' not in group: continue
        a=np.asarray(group['catface']['landmarks'])[:2]
        b=np.asarray(group['catflw']['landmarks'])[[3,1]]
        error=min(np.linalg.norm(a-b,axis=1).mean(),np.linalg.norm(a-b[::-1],axis=1).mean())/max(np.linalg.norm(a[0]-a[1]),1)
        if error>.3:
            disagreements.append(dict(cat=group['catface']['path'],catflw=group['catflw']['path'],normalized_eye_error=float(error)))
            for r in group.values():
                if 'cross_schema_eye_disagreement' not in r['cleaning_flags']:r['cleaning_flags'].append('cross_schema_eye_disagreement')
    save(OUT/'annotation-disagreements.jsonl',disagreements)
    save(cache,result)
    save(DATA/'clean.landmarks-candidates.jsonl',[r for r in result if not r['cleaning_flags']])
    save(OUT/'landmark-review.jsonl',[r for r in result if r['cleaning_flags']])
    print(Counter(f for r in result for f in r['cleaning_flags']),flush=True)
    print('candidates',sum(not r['cleaning_flags'] for r in result),flush=True)
