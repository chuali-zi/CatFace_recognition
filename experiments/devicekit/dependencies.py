"""Explicit online preparation / offline installation with durable reports."""
import argparse,json,platform,subprocess,sys,uuid
from pathlib import Path
from .build import digest
from .run import verify_manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['download','install']); p.add_argument('--wheels',type=Path,required=True)
    p.add_argument('--arch',choices=['aarch64','x86_64']); p.add_argument('--python',choices=['3.10','3.11','3.12']); a=p.parse_args()
    root=Path(__file__).resolve().parents[2]; target=a.wheels.resolve()
    result=root/'results'/('dependencies-'+uuid.uuid4().hex[:8]); result.mkdir(parents=True,mode=0o700)
    report={'action':a.action,'status':'BLOCKED','code':'E_WHEELS','simulated':False,'network':a.action=='download'}
    try:
        if a.action=='download':
            if not a.arch or not a.python: raise ValueError('Explicit target --arch and --python required')
            target.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,'-m','pip','download','--only-binary=:all:','--implementation','cp','--python-version',a.python,'--platform','manylinux_2_28_'+a.arch,'--platform','manylinux_2_27_'+a.arch,'--platform','manylinux_2_17_'+a.arch,'--platform','manylinux2014_'+a.arch,'--dest',str(target),'-r',str(root/'experiments/devicekit/requirements.txt')]
        else:
            if sys.prefix==sys.base_prefix: raise ValueError('Install inside a virtual environment only')
            files=verify_manifest(target,'WHEELS.json'); spec=json.loads((target/'WHEELS.json').read_text())
            if platform.machine()!=spec['arch'] or platform.python_version().rsplit('.',1)[0]!=spec['python']: raise ValueError('Wheel target architecture/Python mismatch')
            actual={str(f.relative_to(target)) for f in target.rglob('*.whl')}
            if actual!=files: raise ValueError('Unsealed/missing wheel')
            command=[sys.executable,'-m','pip','install','--no-index','--only-binary=:all:','--find-links',str(target),'-r',str(root/'experiments/devicekit/requirements.txt')]
        with (result/'private-install.log').open('w') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
        report['exit_code']=rc
        if rc: raise RuntimeError('pip failed; inspect private-install.log locally')
        if a.action=='download':
            (target/'WHEELS.json').write_text(json.dumps({'arch':a.arch,'python':a.python,'files':[{'path':f.name,'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(target.glob('*.whl'))]},indent=2)+'\n')
        report.update(status='PASS',code='OK')
    except Exception as exc: report.update(error_type=type(exc).__name__,reason=str(exc))
    (result/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (result/'report.md').write_text('# Dependencies: '+report['status']+'\n'+report.get('reason','Completed')+'\n')
    print(result); return 0 if report['status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
