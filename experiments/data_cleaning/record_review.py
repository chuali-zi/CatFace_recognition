"""Record an explicitly inspected contact-sheet range, with listed rejections."""
import argparse
import csv
import json
from clean import OUT, WORK, read

parser=argparse.ArgumentParser()
parser.add_argument('start',type=int)
parser.add_argument('stop',type=int,help='Exclusive end index')
parser.add_argument('--reject',default='',help='index:reason,index:reason')
args=parser.parse_args()
rows=read(OUT/'proposals.jsonl')
assert 0<=args.start<args.stop<=len(rows)
reject={int(item.split(':',1)[0]):item.split(':',1)[1] for item in args.reject.split(',') if item}
assert all(args.start<=i<args.stop for i in reject)
path=WORK/'visual-decisions.csv'
old={r['path']:r for r in csv.DictReader(path.open(encoding='utf-8-sig'))} if path.exists() else {}
for i in range(args.start,args.stop):
    p=rows[i]['path']
    if i in reject or p not in old or old[p].get('face_box')!=json.dumps(rows[i]['face']['box']):
        old[p]=dict(path=p,decision='review' if i in reject else 'accept',reason=reject.get(i,'face_and_group_visually_checked'),face_box=json.dumps(rows[i]['face']['box']))
with path.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['path','decision','reason','face_box'])
    writer.writeheader();writer.writerows(old.values())
print('Recorded range',args.start,args.stop,'explicit rejections',len(reject),'decisions',len(old))
