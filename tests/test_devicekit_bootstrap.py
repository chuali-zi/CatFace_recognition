import io
import tarfile
from pathlib import Path

import pytest

from experiments.devicekit.bootstrap import unpack
from experiments.devicekit.resources import verify_manifest


def archive(path, entries):
    with tarfile.open(path,'w:gz') as tar:
        for name,kind,value in entries:
            info=tarfile.TarInfo(name)
            if kind=='link':
                info.type=tarfile.SYMTYPE; info.linkname=value; tar.addfile(info)
            else:
                data=value.encode(); info.size=len(data); info.mode=0o755
                tar.addfile(info,io.BytesIO(data))


def test_runtime_extract_preserves_internal_link_without_execution(tmp_path):
    src=tmp_path/'runtime.tar.gz'; out=tmp_path/'unpack'; out.mkdir()
    archive(src,[('python/bin/python3.11','file','not an executable'),
                 ('python/bin/python3','link','python3.11')])
    try:
        unpack(src,out)
    except OSError as exc:
        if getattr(exc,'winerror',None)==1314: pytest.skip('Windows symlink privilege unavailable')
        raise
    assert (out/'python/bin/python3').read_text()=='not an executable'


@pytest.mark.parametrize('entries',[
    [('../escape','file','bad')],
    [('/python/bin/bad','file','bad')],
    [('python/bin/python3','link','/etc/passwd')],
    [('python/bin','link','../../outside'),('python/bin/child','file','bad')],
    [('python/bin','link','lib'),('python/bin/child','file','bad')],
])
def test_runtime_extract_rejects_escaping_or_link_parent_before_writes(tmp_path,entries):
    src=tmp_path/'bad.tar.gz'; out=tmp_path/'out'; out.mkdir()
    archive(src,entries)
    with pytest.raises(ValueError): unpack(src,out)
    assert list(out.iterdir())==[]


def test_windows_produced_asset_manifest_uses_linux_paths(tmp_path,monkeypatch):
    import json,sys
    from experiments.devicekit.seal_assets import main
    (tmp_path/'detector').mkdir(); (tmp_path/'detector/model').write_bytes(b'model')
    (tmp_path/'provenance.json').write_text(json.dumps({'sources':['research'],'licenses':['retained']}))
    monkeypatch.setattr(sys,'argv',['seal_assets',str(tmp_path)])
    main()
    data=json.loads((tmp_path/'ASSET_MANIFEST.json').read_text())
    assert 'detector/model' in {r['path'] for r in data['files']}
    assert all('\\' not in r['path'] for r in data['files'])
    verify_manifest(tmp_path,'ASSET_MANIFEST.json')


def test_case_collisions_report_filesystem_requirement_before_extraction(tmp_path,monkeypatch):
    import experiments.devicekit.bootstrap as bootstrap
    source=tmp_path/'case.tar.gz'; out=tmp_path/'out'; out.mkdir()
    archive(source,[('python/share/Eterm','file','one'),('python/share/eterm','file','two')])
    monkeypatch.setattr(bootstrap,'case_sensitive',lambda directory:False)
    with pytest.raises(ValueError,match='case-sensitive Linux filesystem'):
        unpack(source,out)
    assert not list(out.iterdir())
