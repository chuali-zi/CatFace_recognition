"""Read-only dataset audit: identity coverage sheets and stronger duplicate search."""
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from catid.prepare import HashIndex

ROOT = Path('data')
OUT = ROOT / 'review' / 'detailed-audit'
OUT.mkdir(parents=True, exist_ok=True)


def read(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding='utf-8').splitlines()]


def thumbnail(row):
    with Image.open(ROOT / row['path']) as im:
        im.draft('RGB', (400, 300))
        return ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (195, 140))


def sheets(name, rows, per_page=36):
    for start in range(0, len(rows), per_page):
        chunk = rows[start:start + per_page]
        canvas = Image.new('RGB', (1200, math.ceil(len(chunk) / 6) * 180), 'white')
        draw = ImageDraw.Draw(canvas)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, (row, im) in enumerate(zip(chunk, pool.map(thumbnail, chunk))):
                x, y = i % 6 * 200, i // 6 * 180
                canvas.paste(im, (x, y))
                draw.text((x + 2, y + 141), f"{start + i}: {row.get('cat_id') or row['source']}", fill='black')
                draw.text((x + 2, y + 156), Path(row['path']).name[:29], fill='black')
        canvas.save(OUT / f'{name}-{start // per_page + 1:03}.jpg', quality=88)
    (OUT / f'{name}.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(name, len(rows), 'images', math.ceil(len(rows) / per_page), 'sheets', flush=True)


def main():
    ready = read('prepared.identity.jsonl')
    conflicts = read('prepared.conflicts.jsonl')
    groups = defaultdict(list)
    for row in ready:
        groups[row['cat_id']].append(row)
    selected = []
    for identity, rows in sorted(groups.items()):
        first = rows[0]
        second = max(rows[1:], key=lambda r: (int(r['dhash'], 16) ^ int(first['dhash'], 16)).bit_count())
        selected.extend([first, second])
    sheets('overview', selected)
    conflict_ids = {r['cat_id'] for r in conflicts if r['task'] == 'identity'}
    affected = [r for r in ready if r['cat_id'] in conflict_ids]
    sheets('conflict-survivors', affected)
    sheets('identity-0138', groups['individuals:0138'])
    index = HashIndex()
    pairs = []
    for i, row in enumerate(ready):
        found = {}
        for field in ('dhash', 'dhash_mirror'):
            for j, distance in index.find(int(row[field], 16), 8):
                found[j] = min(found.get(j, 65), distance)
        for j, distance in found.items():
            pairs.append(dict(path_a=ready[j]['path'], path_b=row['path'],
                              id_a=ready[j]['cat_id'], id_b=row['cat_id'], distance=distance))
        index.add(int(row['dhash'], 16), i)
    pairs.sort(key=lambda p: (p['distance'], p['path_a'], p['path_b']))
    (OUT / 'additional-near-pairs.json').write_text(json.dumps(pairs), encoding='utf-8')
    cross = [p for p in pairs if p['id_a'] != p['id_b']]
    by_path = {r['path']: r for r in ready}
    sheets('cross-id-near', [by_path[p[k]] for p in cross[:72] for k in ('path_a', 'path_b')])
    same = [p for p in pairs if p['id_a'] == p['id_b']]
    sheets('same-id-near', [by_path[p[k]] for p in same[:36] for k in ('path_a', 'path_b')])
    summary = dict(ready_images=len(ready), ready_ids=len(groups), overview_images=len(selected),
                   conflict_affected_ready_images=len(affected), conflict_affected_ready_ids=len({r['cat_id'] for r in affected}),
                   additional_near_pairs=len(pairs), additional_cross_id_pairs=len(cross),
                   near_distance_counts=dict(Counter(p['distance'] for p in pairs)),
                   hsc_views=dict(Counter(Path(r['path']).parent.name for r in ready if r['source']=='hsc')))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
