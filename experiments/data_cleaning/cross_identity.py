"""Show the closest face-hash pairs across provisionally accepted identities."""
import csv
import math
from collections import Counter
from PIL import Image, ImageDraw, ImageOps
from clean import DATA, OUT, WORK, open_image, read, save

decisions={r['path']:r for r in csv.DictReader((WORK/'visual-decisions.csv').open(encoding='utf-8-sig'))}
rows=read(OUT/'proposals.jsonl')
suspect_reasons={'within_identity_visual_conflict','cross_identity_visual_ambiguity','identity_consistency_uncertain'}
suspect={r['cat_id'] for r in rows if decisions[r['path']]['reason'] in suspect_reasons}|{'individuals:0030'}
blocked={r['cat_id'] for r in csv.DictReader((DATA/'manual-group-review.csv').open(encoding='utf-8-sig'))}|suspect
rows=[r for r in rows if decisions[r['path']]['decision']=='accept' and r['cat_id'] not in blocked]
counts=Counter(r['cat_id'] for r in rows)
rows=[r for r in rows if counts[r['cat_id']]>=2]
hashes={r['path']:int(r['phash'],16) for r in read(OUT/'face-hashes.jsonl')}
best={}
for i,a in enumerate(rows):
    for b in rows[i+1:]:
        if a['cat_id']==b['cat_id']:continue
        distance=(hashes[a['path']]^hashes[b['path']]).bit_count()
        key=tuple(sorted([a['cat_id'],b['cat_id']]))
        if key not in best or distance<best[key]['distance']:
            best[key]=dict(a=a,b=b,distance=distance)
pairs=sorted(best.values(),key=lambda p:(p['distance'],p['a']['cat_id'],p['b']['cat_id']))[:100]
save(OUT/'cross-identity-pairs.jsonl',pairs)
for start in range(0,len(pairs),12):
    page=Image.new('RGB',(1440,math.ceil(len(pairs[start:start+12])/3)*185),'white')
    draw=ImageDraw.Draw(page)
    for j,pair in enumerate(pairs[start:start+12]):
        x,y=j%3*480,j//3*185
        for side,k in enumerate(['a','b']):
            r=pair[k]
            crop=ImageOps.contain(open_image(r['path']).crop(r['face']['box']),(225,150))
            page.paste(crop,(x+side*235,y))
            draw.text((x+side*235,y+153),f'{start+j} {r["cat_id"]} d={pair["distance"]}',fill='black')
    page.save(OUT/f'cross-{start//12+1:03}.jpg',quality=93)
print('Compared',len(rows),'accepted images;',len(best),'identity pairs; nearest 100 distances',Counter(p['distance'] for p in pairs))
