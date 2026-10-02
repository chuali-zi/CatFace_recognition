"""Conservative face-data cleaning. Run from the repository root.

Detection proposes crops; only explicitly reviewed identity crops enter clean.identity.
Original files, labels and session IDs are retained without inventing ground truth.
"""
import argparse
import csv
import json
import math
import time
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps

cv2.ocl.setUseOpenCL(False)

DATA = Path('data')
WORK = Path('experiments/data_cleaning')
OUT = DATA / 'review' / 'cleaning'


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def open_image(path):
    with Image.open(DATA / path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def geometry(points, size):
    """Nine CAT keypoints: two eyes, mouth, six ear points; no guessed face box."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    w, h = size
    if len(p) != 9 or not np.isfinite(p).all():
        return None
    lo, hi = p.min(0), p.max(0)
    eye = np.linalg.norm(p[0] - p[1])
    if eye < 16 or (hi-lo).min() < 48:
        return None
    if np.any(p < 0) or np.any(p >= [w, h]):
        return None
    lo = lo - (hi-lo) * [0.10, 0.08]
    hi = hi + (hi-lo) * [0.10, 0.25]
    box = [max(0, int(lo[0])), max(0, int(lo[1])), min(w, int(math.ceil(hi[0]))), min(h, int(math.ceil(hi[1])))]
    return box


def quality(crop):
    gray = cv2.cvtColor(np.asarray(crop.resize((128, 128), Image.Resampling.BILINEAR)), cv2.COLOR_RGB2GRAY)
    return dict(sharpness=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
                contrast=round(float(gray.std()), 2), brightness=round(float(gray.mean()), 2))


def face_hash(row):
    gray=np.asarray(open_image(row['path']).crop(row['face']['box']).convert('L').resize((32,32)),dtype=np.float32)
    low=cv2.dct(gray)[:8,:8].flatten()
    bits=low>np.median(low[1:])
    return dict(path=row['path'],box=row['face']['box'],phash=f'{sum(int(b)<<i for i,b in enumerate(bits)):016x}')


class Detector:
    def __init__(self, threads=4):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(WORK / 'models/ecpose.onnx'), options, providers=['CPUExecutionProvider'])
        self.mean = np.array([.485,.456,.406], dtype=np.float32)
        self.std = np.array([.229,.224,.225], dtype=np.float32)

    def __call__(self, im):
        tensor = np.asarray(im.resize((640,640),Image.Resampling.BILINEAR), dtype=np.float32) / 255
        tensor = ((tensor-self.mean)/self.std).transpose(2,0,1)[None]
        result = self.session.run(None, {'images':tensor, 'orig_target_sizes':np.array([im.size],dtype=np.float32)})
        outputs = {o.name:v[0] for o,v in zip(self.session.get_outputs(), result)}
        detections=[]
        for score,label,points in zip(outputs['scores'],outputs['labels'],outputs['keypoints']):
            if int(label)==1 and float(score)>=.25:
                p=np.asarray(points).reshape(9,-1)[:,:2]
                detections.append(dict(score=round(float(score),5), points=p.tolist(), box=geometry(p,im.size)))
        return sorted(detections,key=lambda r:-r['score'])


class CascadeDetector:
    """Fast proposal generator. Every retained crop still needs visual review."""
    def __init__(self):
        self.cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalcatface_extended.xml')
        cv2.setNumThreads(2)

    def __call__(self, im):
        scale=min(1,720/max(im.size))
        small=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.BILINEAR)
        gray=cv2.cvtColor(np.asarray(small),cv2.COLOR_RGB2GRAY)
        boxes=self.cascade.detectMultiScale(gray,scaleFactor=1.08,minNeighbors=4,minSize=(40,40))
        return [dict(score=None,box=[max(0,int(x/scale)),max(0,int(y/scale)),min(im.width,int((x+w)/scale)),min(im.height,int((y+h)/scale))],
                     method='opencv_frontalcatface_extended') for x,y,w,h in boxes]


def scan(args):
    rows=read(DATA/'prepared.identity.jsonl')
    if args.limit:
        rows=rows[:args.limit]
    cache=OUT/f'detections-{args.detector}.jsonl'
    done={r['path']:r for r in read(cache)} if cache.exists() else {}
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time()
    local=threading.local()
    def inspect(row):
        stat=(DATA/row['path']).stat()
        old=done.get(row['path'])
        if old and old.get('file_size')==stat.st_size and old.get('file_mtime_ns')==stat.st_mtime_ns:
            return None
        if not hasattr(local,'detector'):
            local.detector=Detector(args.threads) if args.detector=='ecpose' else CascadeDetector()
        im=open_image(row['path'])
        detections=local.detector(im)
        for detection in detections:
            if detection['box']:
                detection.update(quality(im.crop(detection['box'])))
        return dict(path=row['path'],cat_id=row['cat_id'],detections=detections,
                    file_size=stat.st_size,file_mtime_ns=stat.st_mtime_ns)
    with cache.open('a',encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=4 if args.detector=='cascade' else 1) as pool:
        for i,result in enumerate(pool.map(inspect,rows),1):
            if result:
                stream.write(json.dumps(result)+'\n');stream.flush()
            if i%100==0 or args.limit:
                print(i,len(rows),round(time.time()-start,1),'seconds',flush=True)


def propose():
    rows=read(DATA/'prepared.identity.jsonl')
    detections={r['path']:r['detections'] for r in read(OUT/'detections-cascade.jsonl')}
    observations=list(csv.DictReader((Path('experiments/data_audit/findings.csv')).open(encoding='utf-8-sig')))
    blocked={r['cat_id'] for r in read(DATA/'prepared.conflicts.jsonl') if r['task']=='identity'}
    blocked.update(r['cat_id'] for r in observations if r['issue']=='multiple_cats_target_unspecified')
    blocked.add('individuals:0138')
    group_file=DATA/'manual-group-review.csv'
    if group_file.exists():
        blocked.update(r['cat_id'] for r in csv.DictReader(group_file.open(encoding='utf-8-sig')))
    bad={r['path']:r['issue'] for r in observations if r['cat_id']}
    eligible,review=[],[]
    for row in rows:
        flags=[]
        ds=detections.get(row['path'])
        if row['cat_id'] in blocked: flags.append('identity_group_target_or_label_unresolved')
        if row['path'] in bad: flags.append(bad[row['path']])
        if ds is None: flags.append('not_scanned')
        elif not ds: flags.append('no_confident_face')
        elif len(ds)>1: flags.append('multiple_face_candidates')
        elif ds[0]['score'] is not None and ds[0]['score']<.75: flags.append('low_face_confidence')
        elif ds[0]['box'] is None: flags.append('invalid_or_small_face_geometry')
        else:
            box=ds[0]['box']
            if min(box[2]-box[0],box[3]-box[1])<96: flags.append('small_face')
            if ds[0]['sharpness']<100: flags.append('soft_face')
            if ds[0]['contrast']<25: flags.append('low_face_contrast')
        out=dict(row,cleaning_flags=flags)
        if ds and len(ds)==1:
            out['face']=ds[0]
        (review if flags else eligible).append(out)
    groups=defaultdict(list)
    hash_path=OUT/'face-hashes.jsonl'
    hashes={r['path']:r for r in read(hash_path)} if hash_path.exists() else {}
    missing=[r for r in eligible if r['path'] not in hashes or hashes[r['path']]['box']!=r['face']['box']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(face_hash,missing): hashes[r['path']]=r
    save(hash_path,hashes.values())
    for row in eligible: groups[row['cat_id']].append(row)
    selected=[]
    for identity,items in sorted(groups.items()):
        # Keep distinct frames per identity; review every retained crop visually.
        ranked=sorted(items,key=lambda r:-min(r['face']['sharpness'],1000))
        picks=[]
        for row in ranked:
            if all(min((int(row['dhash'],16)^int(p['dhash'],16)).bit_count(),
                       (int(row['dhash_mirror'],16)^int(p['dhash'],16)).bit_count())>8 and
                       (int(hashes[row['path']]['phash'],16)^int(hashes[p['path']]['phash'],16)).bit_count()>8 for p in picks):
                picks.append(row)
        if len(picks)>=2: selected.extend(picks)
    save(OUT/'proposals.jsonl',selected)
    save(OUT/'automatic-review.jsonl',review)
    selected_paths={r['path'] for r in selected}
    save(OUT/'unselected.jsonl',[dict(r,cleaning_flags=['similar_frame_or_insufficient_identity_samples']) for r in eligible if r['path'] not in selected_paths])
    sheets(selected)
    print('eligible',len(eligible),'proposed',len(selected),'ids',len({r['cat_id'] for r in selected}),flush=True)
    print(Counter(f for r in review for f in r['cleaning_flags']),flush=True)


def sheets(rows):
    OUT.mkdir(parents=True,exist_ok=True)
    def previews(row):
        im=open_image(row['path'])
        return ImageOps.contain(im.crop(row['face']['box']),(235,155)),ImageOps.contain(im,(110,60))
    for start in range(0,len(rows),24):
        page=Image.new('RGB',(1440,math.ceil(len(rows[start:start+24])/6)*230),'white')
        draw=ImageDraw.Draw(page)
        chunk=rows[start:start+24]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for j,(row,(face,thumb)) in enumerate(zip(chunk,pool.map(previews,chunk))):
                x,y=j%6*240,j//6*230
                page.paste(face,(x,y));page.paste(thumb,(x,y+155))
                draw.text((x+113,y+157),str(start+j),fill='black')
                draw.text((x+113,y+173),row['cat_id'].split(':')[-1][:19],fill='black')
                draw.text((x+2,y+215),Path(row['path']).name[:34],fill='black')
        page.save(OUT/f'faces-{start//24+1:03}.jpg',quality=92)
        print('sheet',start//24+1,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['scan','propose'])
    parser.add_argument('--limit',type=int)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--detector',choices=['cascade','ecpose'],default='cascade')
    args=parser.parse_args()
    scan(args) if args.action=='scan' else propose()
