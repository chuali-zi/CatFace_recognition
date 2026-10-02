"""Materialize the visual observations recorded during the 2026-09-17 audit.

Indices refer to the saved contact-sheet JSON, not the full input manifest.
Unlisted images have NOT been approved. Original manifests are never changed.
"""
import csv
import html
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

OUT = Path('data/review/detailed-audit')
LEDGER = Path('experiments/data_audit/findings.csv')
findings = {}


def record(sheet, indices, issue, note, status='confirmed'):
    rows = json.loads((OUT / f'{sheet}.json').read_text(encoding='utf-8'))
    per_page = 24 if sheet == 'landmark-check' else 36
    for i in indices:
        row = rows[i]
        key = (row['path'], issue)
        findings[key] = dict(path=row['path'], cat_id=row.get('cat_id') or '',
                             issue=issue, status=status, note=note,
                             sheet=f'{sheet}-{i // per_page + 1:03}.jpg', index=i)


multi_note = '同框可见至少两只猫或另一只猫的身体；整图身份标签未指定目标区域，需标目标后再判断能否使用。'
record('overview', [162,170,171,193,206,207,216,218,224,225,226,227,
    238,239,246,247,260,262,263,264,265,274,275,286,287,295,300,301,
    310,311,312,328,329,332,372,373,374,375,384,386,446,447,454,455,
    497,541,542,543,546,547,552,553,568,569,584,585,586,620,621,622,
    623,624,626,627,648,789,885,1010,1011,1016,1022,1023,1026,1027,
    1028,1036,1038,1068,1078,1079,1084,1085], 'multiple_cats_target_unspecified', multi_note)
record('identity-0138', range(12), 'multiple_cats_target_unspecified', multi_note)
record('conflict-survivors', [22,30,42,43,44,45,46,47,48,60,94,95,96,112,
    123,126,127,145,172,173,174,175,*range(176,197),207,208,209,210,211,
    215,216,217,*range(229,243),253,254,255,256,*range(264,279),
    279,280,281,282,283,284,290,294], 'multiple_cats_target_unspecified', multi_note)
record('overview', [199,201,250,1005], 'severe_blur', '明显运动模糊或失焦，猫脸细节不可辨；需按目标区域质量复核。')
record('conflict-survivors', [199,221,222,234,285,286,287], 'severe_blur', '明显运动模糊或失焦，猫脸细节不可辨；需按目标区域质量复核。')
overview = json.loads((OUT / 'overview.json').read_text(encoding='utf-8'))
record('overview', [i for i, r in enumerate(overview) if r['path'].endswith('/0141/0141_000.JPG')], 'no_visible_cat', '已放大原图检查：误拍地面和人的腿，未见猫。')
record('overview', [343], 'occluded_blurred_face', '笼门及门扣遮挡猫脸，且明显失焦；不宜直接作为可用猫脸样本。')
record('overview', [1058], 'no_usable_cat_face', '已放大原图检查：人物占据主体，底部仅有模糊局部猫耳/头部，没有可辨识猫脸。')
record('landmark-check', [96], 'overlapping_landmark_coordinates',
       '第 0 点眼睛与第 3 点耳根均为 (209, 232)；重合已确认，是否为标注错误需核对点位规范和原图。', 'needs_review')

rows = sorted(findings.values(), key=lambda r: (r['issue'], r['path']))
with LEDGER.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
coverage = {}
all_identity_paths = set()
for name in ['overview', 'conflict-survivors', 'identity-0138', 'cross-id-near', 'same-id-near', 'landmark-check']:
    sample = json.loads((OUT / f'{name}.json').read_text(encoding='utf-8'))
    coverage[name] = dict(slots=len(sample), unique_paths=len({r['path'] for r in sample}))
    if name != 'landmark-check':
        all_identity_paths.update(r['path'] for r in sample)
summary = dict(coverage=coverage, visual_identity_unique_paths=len(all_identity_paths),
               finding_rows=len(rows), finding_unique_paths=len({r['path'] for r in rows}),
               issues=dict(Counter(r['issue'] for r in rows)),
               confirmed_identity_paths=len({r['path'] for r in rows if r['status']=='confirmed'}),
               confirmed_identity_ids=len({r['cat_id'] for r in rows if r['status']=='confirmed'}))
(OUT / 'visual-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
parts = ['<!doctype html><meta charset="utf-8"><title>猫数据详细检查 2026-09-17</title>',
         '<style>body{font:16px sans-serif;max-width:1250px;margin:32px auto}img{max-width:100%}td{padding:8px;border-bottom:1px solid #ccc}p{line-height:1.7}</style>',
         '<h1>猫数据详细检查 · 2026-09-17</h1><p>当前 prepared 集合是机械筛选候选集，尚未清理完成。以下是已记录问题的下限，未列出的图不表示通过。多猫同框需标定目标，不直接等同于身份错标。本次未改动原图和候选清单。</p>',
         '<p><a href="../../../docs/data-audit-2026-09-17.md">详细报告</a> · <a href="../../../experiments/data_audit/findings.csv">逐文件问题清单</a></p>',
         '<h2>逐文件问题（点击文件名打开原图）</h2><table><tr><th>身份 / 原图</th><th>问题</th><th>说明</th></tr>']
for r in rows:
    url = '../../' + quote(r['path'])
    parts.append(f'<tr><td>{html.escape(r["cat_id"])}<br><a href="{url}">{html.escape(Path(r["path"]).name)}</a></td><td>{html.escape(r["issue"])}<br>{r["status"]}</td><td>{html.escape(r["note"])} <a href="{r["sheet"]}">检查图 #{r["index"]}</a></td></tr>')
parts.append('</table><h2>已检查的接触表</h2><p>身份总览每个身份选首图与 dHash 差异最大的另一图；不是随机抽样，不能外推总体错误率。缩略图检查不能代替逐张原分辨率身份核验。</p>')
for p in sorted(OUT.glob('*.jpg')):
    parts.append(f'<details><summary>{p.name}</summary><a href="{p.name}"><img loading="lazy" src="{p.name}"></a></details>')
(OUT / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
