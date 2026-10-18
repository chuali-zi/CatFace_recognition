"""Build a complete local research kit from reviewed source and sealed assets."""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from .build import digest
from .resources import verify_manifest

ROOT=Path(__file__).resolve().parents[2]
PARENT_SHA='7143484ec2c1a95c87ea589e7d1aebcd70ea31003d431a8c052a0997ffe30309'


def build(parent_zip, runtime, out):
    if digest(parent_zip)!=PARENT_SHA: raise ValueError('Unexpected parent ZIP; review its provenance first')
    verify_manifest(runtime,'MANIFEST.json')
    out.mkdir(parents=True,exist_ok=False)
    kit=out/'catface-device-research-kit-v3'
    kit.mkdir()
    source_records=[]

    def copy_source(relative):
        source=ROOT/relative
        if not source.is_file() or source.is_symlink(): raise ValueError('Missing/linked reviewed source: '+relative)
        target=kit/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        source_records.append(dict(path=relative,sha256=digest(source),bytes=source.stat().st_size))

    with zipfile.ZipFile(parent_zip) as z:
        for entry in z.infolist():
            name=PurePosixPath(entry.filename)
            if entry.is_dir(): continue
            if name.is_absolute() or '..' in name.parts or name.parts[0]!='catface-device-research-kit':
                raise ValueError('Unsafe parent entry')
            relative=PurePosixPath(*name.parts[1:]).as_posix()
            # Explicit whitelist: no prior result ZIPs, photos, raw data or secrets.
            if relative.startswith(('assets/','licenses/')) or relative in ('LICENSE_REVIEW.zh.md','THIRD_PARTY_NOTICES.md'):
                target=kit/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(entry))
            elif relative.startswith(('src/','experiments/','docs/')) or relative in ('AGENTS.md','pyproject.toml'):
                copy_source(relative)
    for file in sorted((ROOT/'experiments/devicekit').rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts:
            copy_source(file.relative_to(ROOT).as_posix())
    for name in ('board-test-2026-10-18.md','device-test-results.md'):
        copy_source('docs/'+name)
    for name in ('run.sh','QUICKSTART.md','MANUAL.zh.md','AGENT.md','ACCEPTANCE.md'):
        shutil.copy2(ROOT/'experiments/devicekit'/name,kit/name)
    shutil.copy2(ROOT/'experiments/devicekit/requirements.txt',kit/'requirements-device.txt')
    (kit/'README.md').write_text('# CatFace v3 ARM64科研测试包\n\n先读 QUICKSTART.md。'
        '含冻结真实模型、ARM64 CPython3.11独立运行时及离线wheels；不含照片或图库。'
        '2026-10-18板端已在已有Python3.11环境通过两模型加载，图片识别未执行；'
        '本修订修复离线解包时符号链接父目录缺失，包内ARM离线安装仍待同学复测。'
        '不控制门，不替换系统Python，不自动打开相机。详见 MANUAL.zh.md、AGENT.md 和 docs/board-test-2026-10-18.md。\n',encoding='utf-8')
    support=kit/'runtime-support'; support.mkdir()
    runtime_spec=json.loads((runtime/'MANIFEST.json').read_text(encoding='utf-8'))
    for entry in runtime_spec['files']:
        relative=entry['path']; target=support/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(runtime/relative,target)
    shutil.copy2(runtime/'MANIFEST.json',support/'MANIFEST.json')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=normal','--']+
        sorted({r['path'] for r in source_records}),cwd=ROOT,text=True).strip())
    save=lambda name,value:(kit/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('BUILD.json',dict(kit_version=3,source_commit=head,working_tree_changes_included=dirty,
        kind='noncommercial-academic-research-test-kit',models_included=True,wheels_included=True,
        python_runtime_included=True,hardware_validated=False,parent_zip_sha256=PARENT_SHA,
        runtime='Linux ARM64 CPython3.11.17, glibc>=2.28; target execution not yet validated'))
    save('SOURCE_MANIFEST.json',dict(source_commit=head,parent_zip_sha256=PARENT_SHA,
        imported_devicekit_source='Sealed parent ZIP; reviewed additive restoration, then local patches',
        current_source_files=list({r['path']:r for r in source_records}.values()),
        excluded='Raw photos, galleries, .env/secrets, old acceptance/result ZIPs, caches and tests'))
    entries=[dict(path=f.relative_to(kit).as_posix(),bytes=f.stat().st_size,sha256=digest(f))
             for f in sorted(kit.rglob('*')) if f.is_file()]
    save('MANIFEST.json',dict(algorithm='sha256',files=entries,note='Byte seal, not board or recognition acceptance'))
    verify_manifest(kit,'MANIFEST.json')
    archive=out/'CatFace-v3-ARM64-Python311-完整测试包.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(kit.rglob('*')):
            if file.is_file(): z.write(file,(Path(kit.name)/file.relative_to(kit)).as_posix())
    (out/'SHA256SUMS.txt').write_text(digest(archive)+'  '+archive.name+'\n',encoding='utf-8')
    shutil.copy2(kit/'MANIFEST.json',out/'内容清单.json')
    print(archive); print(digest(archive))
    return archive


def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(); p.add_argument('--parent-zip',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); build(a.parent_zip.resolve(),a.runtime.resolve(),a.out.resolve())


if __name__=='__main__': main()
