"""Build a source/readiness ZIP; never manufacture absent research assets."""
import argparse, hashlib, json, shutil, subprocess, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    out=a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    files+= [str(f.relative_to(ROOT)) for f in (ROOT/'experiments/devicekit').rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    for name in sorted(set(files)):
        source=ROOT/name
        if source.is_file():
            target=out/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    for name in ('run.sh','QUICKSTART.md','MANUAL.zh.md','AGENT.md','ACCEPTANCE.md'):
        shutil.copy2(ROOT/'experiments/devicekit'/name,out/name)
    shutil.copy2(ROOT/'experiments/devicekit/requirements.txt',out/'requirements-device.txt')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (out/'BUILD.json').write_text(json.dumps({'source_commit':commit,'working_tree_changes_included':bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()),'kind':'installation-and-readiness-kit','kit_version':2,'models_included':False,'wheels_included':False,'hardware_validated':False},indent=2)+'\n')
    entries=[{'path':f.relative_to(out).as_posix(),'sha256':digest(f),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file()]
    (out/'MANIFEST.json').write_text(json.dumps({'algorithm':'sha256','files':entries},indent=2)+'\n')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file(): z.write(f,Path(out.name)/f.relative_to(out))
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print(archive); print(digest(archive))
if __name__=='__main__': main()
