"""Validate the actual release files and complete source disposition."""
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from clean import DATA, OUT, WORK, open_image, read

original=read(DATA/'samples.expanded.jsonl')
identity=read(DATA/'clean.identity.jsonl')
landmarks=read(DATA/'clean.landmarks.jsonl')
review=read(DATA/'clean.review.jsonl')
duplicates=read(DATA/'clean.duplicates.jsonl')
blocked={r['cat_id'] for r in csv.DictReader((DATA/'manual-group-review.csv').open(encoding='utf-8-sig'))}
manual={r['path'] for r in csv.DictReader((DATA/'manual-review.csv').open(encoding='utf-8-sig'))}
decisions={r['path']:r for r in csv.DictReader((WORK/'visual-decisions.csv').open(encoding='utf-8-sig'))}
source={r['path']:r for r in original}
parts=[{r['original_path'] for r in identity}, {r['path'] for r in landmarks},
       {r['path'] for r in review}, {r['path'] for r in duplicates}]
assert sum(len(p) for p in parts)==len(original)==len(source)
assert all(not a&b for i,a in enumerate(parts) for b in parts[i+1:])
assert set.union(*parts)==set(source)
assert not parts[0]&manual
assert not {r['cat_id'] for r in identity}&blocked
counts=Counter(r['cat_id'] for r in identity)
assert all(2<=n<=12 for n in counts.values())
assert len({r['pixel_hash'] for r in identity})==len(identity)
def verify_face(r):
    assert decisions[r['original_path']]['decision']=='accept'
    assert json.loads(decisions[r['original_path']]['face_box'])==r['face_box']
    assert r['cat_id']==source[r['original_path']]['cat_id']
    assert r['session_id'] is None and r['leakage_group']==r['cat_id']
    assert r['review_status']=='agent_visual_checked'
    with Image.open(DATA/r['path']) as im:
        im.load()
        assert im.mode=='RGB' and im.size==(r['width'],r['height'])
        assert min(im.size)>=96
        pixels=im.tobytes()
        digest=hashlib.sha256(f'{im.width}x{im.height}:'.encode()+pixels).hexdigest()
        assert digest==r['pixel_hash']
        expected=open_image(r['original_path']).crop(r['face_box'])
        assert im.size==expected.size and pixels==expected.tobytes()
    return (DATA/r['path']).stat().st_size
with ThreadPoolExecutor(max_workers=4) as pool:
    sizes=list(pool.map(verify_face,identity))
assert {p.relative_to(DATA).as_posix() for p in (DATA/'clean/faces').rglob('*.png')}=={r['path'] for r in identity}
for r in landmarks:
    assert r['cat_id'] is None and r['task']=='landmarks'
    assert r['annotation_review_status']=='pending_semantic_review'
    assert (DATA/r['path']).is_file()
    assert len(r['landmarks'])=={'catface':9,'catflw':48}[r['source']]
    assert len(set(map(tuple,r['landmarks'])))==len(r['landmarks'])
    assert all(0<=x<r['width'] and 0<=y<r['height'] for x,y in r['landmarks'])
proposals=read(OUT/'proposals.jsonl')
assert {r['path'] for r in proposals}<=decisions.keys()
cross_review=read(OUT/'cross-identity-review.jsonl')
for r in cross_review:
    if r['decision']=='quarantine_both_identity_groups':
        assert {r['cat_id_a'],r['cat_id_b']}<=blocked
result=dict(input_records=len(original),identity_images=len(identity),identity_ids=len(counts),
            identity_count_range=[min(counts.values()),max(counts.values())],identity_png_bytes=sum(sizes),
            checked_every_exported_png=True,checked_lossless_crop_against_original=True,
            exact_crop_duplicates=0,excluded_group_leaks=0,manual_exclusion_leaks=0,
            disposition_partition_complete=True,disposition_overlap=0,
            visual_proposal_decisions=len(proposals),cross_identity_pairs_visually_compared=len(cross_review),
            landmark_geometry_candidates=len(landmarks),landmark_semantic_review='pending')
(DATA/'clean.validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
