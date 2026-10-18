"""Build a complete local research kit from reviewed source and sealed assets."""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from .build import digest
from .resources import verify_manifest, verify_assets

ROOT=Path(__file__).resolve().parents[2]
PARENT_SHA='7143484ec2c1a95c87ea589e7d1aebcd70ea31003d431a8c052a0997ffe30309'


def build(parent_zip, runtime, out, historical_assets=None):
    if digest(parent_zip)!=PARENT_SHA: raise ValueError('Unexpected parent ZIP; review its provenance first')
    verify_manifest(runtime,'MANIFEST.json')
    out.mkdir(parents=True,exist_ok=False)
    kit=out/('catface-device-research-kit-v4' if historical_assets else 'catface-device-research-kit-v3')
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
    for name in ('board-test-2026-10-18.md','device-test-results.md','handoff-2026-10-18.md','handoff-assets.json','handoff-validation-2026-10-18.md'):
        copy_source('docs/'+name)
    for file in sorted((ROOT/'docs/results/board-2026-10-18').glob('*.json')):
        copy_source(file.relative_to(ROOT).as_posix())
    for name in ('run.sh','run-all.sh','QUICKSTART.md','MANUAL.zh.md','AGENT.md','ACCEPTANCE.md'):
        shutil.copy2(ROOT/'experiments/devicekit'/name,kit/name)
    shutil.copy2(ROOT/'experiments/devicekit/requirements.txt',kit/'requirements-device.txt')
    if historical_assets:
        verify_assets(historical_assets, 'historical', ['small160', 'large224'])
        manifest=json.loads((historical_assets/'ASSET_MANIFEST.json').read_text(encoding='utf-8'))
        for entry in manifest['files']:
            relative=entry['path']; target=kit/'assets'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(historical_assets/relative,target)
        shutil.copy2(historical_assets/'ASSET_MANIFEST.json',kit/'assets/ASSET_MANIFEST.json')
        verify_assets(kit/'assets','historical',['small160','large224'])
        provenance_path=kit/'assets/provenance.json'
        provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
        provenance.update(distribution_scope='User-requested private project handoff for classmate noncommercial research; no public upload or direct third-party send performed',
            historical_data_included=True,
            historical_data_source='Cat Individual Images; existing data-sources.md records CC BY 4.0; publisher identities, no field truth',
            historical_data_attribution='timost1234, Cat Individual Images, https://www.kaggle.com/datasets/timost1234/cat-individuals; existing images copied without pixel changes, filenames anonymized')
        provenance_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for entry in manifest['files']:
            if entry['path']=='provenance.json':
                entry.update(sha256=digest(provenance_path),bytes=provenance_path.stat().st_size)
        (kit/'assets/ASSET_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        notice='\n\n## v4科研交接补充（2026-10-18）\n\n本次用户要求项目整体交接给对应同学。v4独立私下交接包增加149张已用于development的原图、两套冻结图库和校准记录；不公开上传GitHub。图片来源为timost1234的Cat Individual Images： https://www.kaggle.com/datasets/timost1234/cat-individuals ，本项目已有data-sources.md记录许可为CC BY 4.0： https://creativecommons.org/licenses/by/4.0/ 。原图像素未修改，包内改用序号文件名；标签沿用发布者身份，不是现场真值。前文“无照片/图库”描述针对v3模型包。上游模型及数据条款继续适用，不扩大为公开/商业授权。\n'
        for name in ('LICENSE_REVIEW.zh.md','THIRD_PARTY_NOTICES.md'):
            with (kit/name).open('a',encoding='utf-8') as handle:
                handle.write(notice)
        verify_assets(kit/'assets','historical',['small160','large224'])
    (kit/'README.md').write_text('# CatFace ARM64科研测试包\n\n先读 QUICKSTART.md。'
        '含冻结真实模型、ARM64 CPython3.11独立运行时及离线wheels；不含照片或图库。'
        if not historical_assets else '# CatFace v4 完整板端测试包\n\n执行 bash run-all.sh。'
        '含149张冻结development原图、两套研究图库、模型和已验证安装的ARM离线运行时。'
        '自动完成两模型×三线程×三轮测量，回传一个结果ZIP。'
        '该图片池已在开发阶段使用，不能当成新现场验收。\n',encoding='utf-8')
    with (kit/'README.md').open('a',encoding='utf-8') as readme:
        readme.write('2026-10-18板端已通过包内离线安装及两模型CPU加载，图片识别尚待板端执行。'
        '不控制门，不替换系统Python，不自动打开相机。详见 MANUAL.zh.md、AGENT.md 和 docs/board-test-2026-10-18.md。\n')
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
    save('BUILD.json',dict(kit_version=4 if historical_assets else 3,source_commit=head,working_tree_changes_included=dirty,
        kind='noncommercial-academic-research-test-kit',models_included=True,wheels_included=True,
        python_runtime_included=True,hardware_validated=False,parent_zip_sha256=PARENT_SHA,
        historical_queries_included=149 if historical_assets else 0,
        runtime='Linux ARM64 CPython3.11.17, glibc>=2.28; board offline install/import/model load passed; image replay pending'))
    save('SOURCE_MANIFEST.json',dict(source_commit=head,parent_zip_sha256=PARENT_SHA,
        imported_devicekit_source='Sealed parent ZIP; reviewed additive restoration, then local patches',
        current_source_files=list({r['path']:r for r in source_records}.values()),
        excluded='.env/secrets, private field data, old result ZIPs, caches and tests; v4 includes only sealed development images/galleries'))
    entries=[dict(path=f.relative_to(kit).as_posix(),bytes=f.stat().st_size,sha256=digest(f))
             for f in sorted(kit.rglob('*')) if f.is_file()]
    save('MANIFEST.json',dict(algorithm='sha256',files=entries,note='Byte seal, not board or recognition acceptance'))
    verify_manifest(kit,'MANIFEST.json')
    archive=out/('CatFace-v4-完整板端测试包.zip' if historical_assets else 'CatFace-v3-ARM64-Python311-完整测试包.zip')
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
    p.add_argument('--historical-assets',type=Path,help='Reviewed sealed historical assets with 149 images and frozen galleries')
    a=p.parse_args(); build(a.parent_zip.resolve(),a.runtime.resolve(),a.out.resolve(),
        a.historical_assets.resolve() if a.historical_assets else None)


if __name__=='__main__': main()
