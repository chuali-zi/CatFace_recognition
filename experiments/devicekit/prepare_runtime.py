"""Prepare pinned ARM64 runtime bytes on an online host; never execute them."""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

RELEASE='20261013'
ARCHIVE='cpython-3.11.17+20261013-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz'
BASE='https://github.com/astral-sh/python-build-standalone/releases/download/'+RELEASE+'/'


def download(url, target):
    with urllib.request.urlopen(url,timeout=60) as source, target.open('xb') as output:
        while True:
            block=source.read(1024*1024)
            if not block: break
            output.write(block)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def seal(out):
    """Verify against upstream SHA256 before sealing, including every PyPI wheel."""
    sums=(out/'SHA256SUMS').read_text(encoding='utf-8')
    expected=next(line.split()[0] for line in sums.splitlines() if line.split()[-1].lstrip('*')==ARCHIVE)
    if digest(out/ARCHIVE) != expected: raise ValueError('Upstream runtime SHA256 mismatch')
    pins=Path(__file__).with_name('requirements.txt').read_text().splitlines()
    lock=[]
    wheel_sources=[]
    for pin in pins:
        name,version=pin.split('==')
        with urllib.request.urlopen('https://pypi.org/pypi/'+name+'/'+version+'/json',timeout=30) as r:
            metadata=json.load(r)
        available={row['filename']:row for row in metadata['urls']}
        candidates=[p for p in (out/'wheels').glob('*.whl') if p.name in available]
        if len(candidates)!=1: raise ValueError('Expected exactly one target wheel for '+pin)
        path=candidates[0]; row=available[path.name]
        if digest(path)!=row['digests']['sha256']: raise ValueError('PyPI wheel SHA256 mismatch')
        lock.append(pin+' --hash=sha256:'+digest(path))
        wheel_sources.append(dict(file=path.name,url=row['url'],sha256=digest(path)))
    (out/'requirements-lock.txt').write_text('\n'.join(lock)+'\n',encoding='utf-8')
    (out/'RUNTIME.json').write_text(json.dumps(dict(python='3.11.17',architecture='aarch64',
        libc_minimum='2.28',archive=ARCHIVE,url=BASE+ARCHIVE,sha256=expected,
        upstream_checksums_url=BASE+'SHA256SUMS',wheels=wheel_sources,
        validation='Downloaded and hashed on Windows; not executed on ARM; board bootstrap/import/model checks required.'),indent=2)+'\n',encoding='utf-8')
    entries=[dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=digest(f))
             for f in sorted(out.rglob('*')) if f.is_file() and f.name!='MANIFEST.json']
    (out/'MANIFEST.json').write_text(json.dumps(dict(algorithm='sha256',files=entries),indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seal-existing',action='store_true',help='Verify already downloaded bytes; no replacement')
    a=p.parse_args(); out=a.out.resolve()
    if not a.seal_existing:
        out.mkdir(parents=True,exist_ok=False)
        download(BASE+'SHA256SUMS',out/'SHA256SUMS')
        download(BASE+ARCHIVE,out/ARCHIVE)
        subprocess.run([sys.executable,'-m','pip','download','--only-binary=:all:','--no-deps',
            '--implementation','cp','--python-version','3.11','--abi','cp311',
            '--platform','manylinux_2_27_aarch64','--platform','manylinux_2_28_aarch64',
            '--platform','manylinux2014_aarch64','--index-url','https://pypi.org/simple',
            '--dest',str(out/'wheels'),'-r',str(Path(__file__).with_name('requirements.txt'))],check=True)
    seal(out)
    print(out)


if __name__=='__main__': main()
