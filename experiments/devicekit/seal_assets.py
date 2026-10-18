"""Seal prepared local asset bytes and source records; do not grant distribution rights."""
import argparse, json
from pathlib import Path
from .build import digest

def main():
    p=argparse.ArgumentParser(); p.add_argument('assets',type=Path); a=p.parse_args(); root=a.assets.resolve()
    provenance=json.loads((root/'provenance.json').read_text())
    for key in ('sources','licenses'):
        if not provenance.get(key): raise ValueError('Missing provenance: '+key)
    if (root/'ASSET_MANIFEST.json').exists(): raise ValueError('Manifest exists; use a new directory for a revised asset set')
    files=[]
    for f in sorted(root.rglob('*')):
        if f.is_symlink(): raise ValueError('Symlinks forbidden')
        if f.is_file(): files.append({'path':f.relative_to(root).as_posix(),'sha256':digest(f),'bytes':f.stat().st_size})
    (root/'ASSET_MANIFEST.json').write_text(json.dumps({'files':files,'note':'Local byte-integrity seal; not independent proof of provenance, legal clearance or distribution permission'},indent=2)+'\n')
    print('Asset manifest SHA256:',digest(root/'ASSET_MANIFEST.json'))
if __name__=='__main__': main()
