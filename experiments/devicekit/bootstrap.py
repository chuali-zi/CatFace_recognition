"""Install the sealed ARM64 runtime locally; executable with system Python 3.8."""
import hashlib
import json
import os
import platform
import posixpath
import shutil
import subprocess
import sys
import tarfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from .resources import verify_manifest

ROOT = Path(__file__).resolve().parents[2]


def validate_members(members):
    links = {m.name.rstrip('/') for m in members if m.issym()}
    seen = set()
    for m in members:
        p = PurePosixPath(m.name)
        if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'python':
            raise ValueError('Unsafe runtime archive path')
        if m.name in seen or any(str(parent) in links for parent in p.parents):
            raise ValueError('Duplicate path or symlink parent')
        seen.add(m.name)
        if not (m.isdir() or m.isfile() or m.issym()):
            raise ValueError('Unsupported runtime archive member')
        if m.issym():
            target = posixpath.normpath(posixpath.join(posixpath.dirname(m.name), m.linkname))
            if PurePosixPath(m.linkname).is_absolute() or not target.startswith('python/'):
                raise ValueError('Escaping runtime symlink')


def case_sensitive(directory):
    probe=directory/('.catface-case-'+uuid.uuid4().hex)
    try:
        probe.write_bytes(b'')
        return not probe.with_name(probe.name.upper()).exists()
    finally:
        if probe.exists(): probe.unlink()


def unpack(archive, destination):
    """Reject escaping paths, special files and writes through archive symlinks."""
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        validate_members(members)
        if len({m.name.casefold() for m in members}) != len(members) and not case_sensitive(destination):
            raise ValueError('Runtime requires a case-sensitive Linux filesystem. Unpack the kit in the board Linux home/ext4, not NTFS/FAT/shared Windows storage.')
        for m in members:
            target = destination.joinpath(*PurePosixPath(m.name).parts)
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(m.mode & 0o777)
        for m in members:
            if m.issym():
                destination.joinpath(*PurePosixPath(m.name).parts).symlink_to(m.linkname)


def main():
    # Only package-local writes; no network, apt, sudo or system Python changes.
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-bootstrap-'+uuid.uuid4().hex[:8]
    out = ROOT/'results'/run_id
    out.mkdir(parents=True, mode=0o700)
    private = out/'private'
    private.mkdir(mode=0o700)
    report = {'schema_version':3, 'mode':'bootstrap', 'run_id':run_id,
              'environment':{'system':platform.system(),'architecture':platform.machine(),
                             'python':platform.python_version(),'libc':list(platform.libc_ver())},
              'stages':[], 'network_used':False, 'system_python_modified':False,
              'model_inference_executed':False, 'hardware_validated':False}

    def stage(status, code, reason):
        report['stages'].append({'stage':'runtime_setup','status':status,'code':code,'reason':reason})

    try:
        if platform.system() != 'Linux' or platform.machine() not in ('aarch64','arm64'):
            raise ValueError('This offline runtime is Linux ARM64 only; no installation attempted.')
        libc = platform.libc_ver()
        if libc[0] != 'glibc' or tuple(map(int,libc[1].split('.')[:2])) < (2,28):
            raise ValueError('Included wheels require glibc >=2.28; system libc will not be changed.')
        verify_manifest(ROOT, 'MANIFEST.json')
        support = ROOT/'runtime-support'
        verify_manifest(support, 'MANIFEST.json')
        spec = json.loads((support/'RUNTIME.json').read_text(encoding='utf-8'))
        target = ROOT/'.runtime'
        if target.exists():
            raise ValueError('.runtime already exists; use it or choose a fresh unpack directory. No overwrite.')
        temporary = ROOT/('.runtime-install-'+uuid.uuid4().hex[:8])
        temporary.mkdir(mode=0o700)
        unpack(support/spec['archive'], temporary)
        interpreter = temporary/'python/bin/python3'
        clean_env = {k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','PIP_TARGET','PIP_PREFIX','PIP_USER')}
        clean_env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull)
        with (private/'install.log').open('w',encoding='utf-8') as log:
            command=[str(interpreter),'-I','-m','pip','install','--no-index','--only-binary=:all:',
                     '--require-hashes','--find-links',str(support/'wheels'),'-r',str(support/'requirements-lock.txt')]
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=clean_env,timeout=600).returncode
            if rc: raise RuntimeError('Offline pip failed, exit '+str(rc)+'; private/install.log retained.')
            check=subprocess.run([str(interpreter),'-I','-c',
                'import json,platform,numpy,PIL,onnxruntime,psutil; '
                'assert "CPUExecutionProvider" in onnxruntime.get_available_providers(); '
                'print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,Pillow=PIL.__version__,onnxruntime=onnxruntime.__version__,psutil=psutil.__version__,providers=onnxruntime.get_available_providers())))'],
                stdout=subprocess.PIPE,stderr=log,text=True,env=clean_env,timeout=60)
            if check.returncode: raise RuntimeError('Runtime import/provider check failed; private/install.log retained.')
        report['installed_environment']=json.loads(check.stdout)
        temporary.rename(target)
        stage('PASS','OK','Package-local .runtime installed offline; run bash run.sh --mode preflight, then model-check. No model inference yet.')
    except Exception as exc:
        with (private/'failure.log').open('w',encoding='utf-8') as log:
            log.write(type(exc).__name__+': '+str(exc)+'\n')
        stage('BLOCKED','E_RUNTIME_SETUP',str(exc))
        report['error_type']=type(exc).__name__
    report['overall']='PASS' if all(s['status']=='PASS' for s in report['stages']) else 'BLOCKED'
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'events.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in report['stages']),encoding='utf-8')
    (out/'report.md').write_text('# CatFace 离线运行时准备\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('report.json','report.md','events.jsonl'): z.write(out/name,name)
    archive.with_suffix('.zip.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    print('Return only:',archive)
    return 0 if report['overall']=='PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
