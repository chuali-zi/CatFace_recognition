"""Export reviewed identity face crops and an exhaustive disposition of source rows."""
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from clean import DATA, OUT, WORK, open_image, read, save


def main():
    proposals=read(OUT/'proposals.jsonl')
    decisions=list(csv.DictReader((WORK/'visual-decisions.csv').open(encoding='utf-8-sig')))
    proposal_paths={r['path'] for r in proposals}
    by_path={r['path']:r for r in decisions if r['path'] in proposal_paths}
    if len({r['path'] for r in decisions})!=len(decisions) or set(by_path)!=proposal_paths:
        raise ValueError('Every proposed image needs exactly one explicit visual decision.')
    if any(r['decision'] not in ('accept','review') for r in decisions):
        raise ValueError('Unsupported visual decision')
    if any(json.loads(by_path[r['path']].get('face_box') or 'null')!=r['face']['box'] for r in proposals):
        raise ValueError('Crop changed since visual review; inspect and record the new crop first.')
    blocked={r['cat_id']:r['reason'] for r in csv.DictReader((DATA/'manual-group-review.csv').open(encoding='utf-8-sig'))}
    accepted=[r for r in proposals if by_path[r['path']]['decision']=='accept' and r['cat_id'] not in blocked]
    counts=Counter(r['cat_id'] for r in accepted)
    accepted=[r for r in accepted if counts[r['cat_id']]>=2]
    # Bound domination by long bursts; preserve diversity among already reviewed crops.
    hashes={r['path']:int(r['phash'],16) for r in read(OUT/'face-hashes.jsonl')}
    groups=defaultdict(list)
    for row in accepted: groups[row['cat_id']].append(row)
    selected=[]
    limited=set()
    for identity,items in sorted(groups.items()):
        remaining=sorted(items,key=lambda r:(-r['face']['sharpness'],r['path']))
        picks=[remaining.pop(0)]
        while remaining and len(picks)<12:
            choice=max(remaining,key=lambda r:(min((hashes[r['path']]^hashes[p['path']]).bit_count() for p in picks),r['face']['sharpness']))
            picks.append(choice);remaining.remove(choice)
        selected.extend(picks)
        limited.update(r['path'] for r in remaining)
    def export_crop(row):
        crop=open_image(row['path']).crop(row['face']['box'])
        relative=Path('clean/faces')/row['source']/row['cat_id'].split(':',1)[1]/(Path(row['path']).stem+'.png')
        destination=DATA/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        crop.save(destination)
        return dict(path=relative.as_posix(),original_path=row['path'],source=row['source'],task='identity',
                          cat_id=row['cat_id'],session_id=None,review_status='agent_visual_checked',
                          identity_label_status='publisher_label_visual_consistency_checked',
                          original_pixel_hash=row['pixel_hash'],width=crop.width,height=crop.height,
                          pixel_hash=hashlib.sha256(f'{crop.width}x{crop.height}:'.encode()+crop.tobytes()).hexdigest(),
                          face_box=row['face']['box'],face_quality={k:row['face'][k] for k in ('sharpness','contrast','brightness')},
                          leakage_group=row['cat_id'],landmarks=None)
    with ThreadPoolExecutor(max_workers=4) as pool:
        generated=list(pool.map(export_crop,selected))
    ready=generated
    # Separate exact crop collisions across labels. Never pick an arbitrary winning ID.
    hashes={}
    for r in ready: hashes.setdefault(r['pixel_hash'],set()).add(r['cat_id'])
    ready=[r for r in ready if len(hashes[r['pixel_hash']])==1]
    counts=Counter(r['cat_id'] for r in ready)
    ready=[r for r in ready if counts[r['cat_id']]>=2]
    unique=[]
    seen=set()
    for row in ready:
        if row['pixel_hash'] not in seen:
            unique.append(row);seen.add(row['pixel_hash'])
    counts=Counter(r['cat_id'] for r in unique)
    ready=[r for r in unique if counts[r['cat_id']]>=2]
    kept={r['path'] for r in ready}
    face_root=(DATA/'clean/faces').resolve()
    for path in face_root.rglob('*.png'):
        if path.relative_to(DATA.resolve()).as_posix() not in kept:
            target=path.resolve()
            assert target.is_relative_to(face_root)
            target.unlink()
    save(DATA/'clean.identity.jsonl',ready)

    landmarks=read(DATA/'clean.landmarks-candidates.jsonl')
    # All keep their original annotation schema and explicit pending semantic-review status.
    save(DATA/'clean.landmarks.jsonl',landmarks)
    identity_paths={r['original_path'] for r in ready}
    landmark_paths={r['path'] for r in landmarks}
    reason={}
    for name in ['automatic-review','unselected','landmark-review']:
        for row in read(OUT/f'{name}.jsonl'):
            reason[row['path']]=row['cleaning_flags']
    for row in proposals:
        if row['path'] not in identity_paths:
            decision=by_path[row['path']]
            reason[row['path']]=[decision['reason']] if decision['decision']=='review' else ['fewer_than_two_accepted_images_or_crop_duplicate']
            if row['cat_id'] in blocked and decision['decision']=='accept':
                reason[row['path']]=blocked[row['cat_id']].split(';')
            if row['path'] in limited: reason[row['path']]=['representative_limit_12_per_identity']
    for name in ['review','conflicts','singletons','unreadable']:
        for row in read(DATA/f'prepared.{name}.jsonl'):
            reason.setdefault(row['path'],row.get('preparation_flags') or [row.get('exclusion_reason',name)])
    duplicates={r['path']:r for r in read(DATA/'prepared.duplicates.jsonl')}
    review=[]
    all_rows=read(DATA/'samples.expanded.jsonl')
    for row in all_rows:
        if row['path'] not in identity_paths|landmark_paths|duplicates.keys():
            flags=[part for flag in reason.get(row['path'],['not_approved_for_this_release']) for part in flag.split(';')]
            if row['task']=='identity' and row['cat_id'] in blocked:
                flags=sorted(set(flags+blocked[row['cat_id']].split(';')))
            review.append(dict(row,cleaning_flags=flags))
    save(DATA/'clean.review.jsonl',review)
    save(DATA/'clean.duplicates.jsonl',duplicates.values())
    assert len(identity_paths)+len(landmark_paths)+len(review)+len(duplicates)==len(all_rows)
    partitions=[identity_paths,landmark_paths,{r['path'] for r in review},set(duplicates)]
    assert all(not a&b for i,a in enumerate(partitions) for b in partitions[i+1:])
    assert set.union(*partitions)=={r['path'] for r in all_rows}
    summary=dict(input_records=len(all_rows),identity_images=len(ready),identity_ids=len({r['cat_id'] for r in ready}),
                 identity_sources=dict(Counter(r['source'] for r in ready)),
                 landmark_records=len(landmarks),landmark_unique_pixels=len({r['pixel_hash'] for r in landmarks}),
                 review_records=len(review),exact_duplicate_records=len(duplicates),
                 visual_proposals=len(proposals),visual_rejections=sum(r['decision']=='review' for r in by_path.values()),
                 identity_group_exclusions=len(blocked),representative_limit=12,representative_limit_exclusions=len(limited),
                 visually_accepted_then_group_excluded=sum(by_path[r['path']]['decision']=='accept' and r['cat_id'] in blocked for r in proposals),
                 review_reasons=dict(Counter(f for r in review for f in r['cleaning_flags'])),
                 identity_status='Every exported crop visually checked by the agent; publisher IDs retained; no field identity ground truth.',
                 landmark_status='Geometry and face-quality checked; point-by-point semantic review pending.',
                 session_status='Unknown. leakage_group is conservatively the whole identity, not an observed session.')
    (DATA/'clean.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    sections=['<!doctype html><meta charset="utf-8"><title>清理结果</title>',
              '<style>body{font:16px sans-serif;max-width:1300px;margin:32px auto}section{display:flex;flex-wrap:wrap}figure{width:190px;margin:8px}img{width:190px;height:170px;object-fit:contain}figcaption{font-size:12px;word-wrap:break-word}</style>',
              f'<h1>身份数据清理结果</h1><p>{len(ready)} 张 / {summary["identity_ids"]} 个身份。已查看每张输出裁剪与原图缩略图，仍沿用发布者身份；没有现场身份真值或真实拍摄场次。</p>',
              '<p>关键点记录仅通过几何、尺寸和清晰度检查，逐点语义审核仍未完成。详细数量见 clean.summary.json。</p>']
    for identity in sorted({r['cat_id'] for r in ready}):
        sections.append(f'<h2>{html.escape(identity)}</h2><section>')
        for row in ready:
            if row['cat_id']!=identity:continue
            url='../../'+quote(row['path'])
            original='../../'+quote(row['original_path'])
            sections.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}"></a><figcaption><a href="{original}">{html.escape(Path(row["original_path"]).name)} 原图</a></figcaption></figure>')
        sections.append('</section>')
    (OUT/'index.html').write_text('\n'.join(sections),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
