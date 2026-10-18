"""Stdlib supervisor: preserve evidence even when the inference dependencies are absent."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from .build import digest
from .resources import safe_file, verify_manifest, verify_assets, verify_local_gallery, plan, MODEL_MODES, HISTORY_MODES
from .platform_check import assess, inventory, error_summary

ROOT = Path(__file__).resolve().parents[2]
PINS = {'numpy':'1.26.4','Pillow':'11.1.0','onnxruntime':'1.20.1','psutil':'6.1.1'}


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def environment():
    info={'system':platform.system(),'release':platform.release(),'architecture':platform.machine(),
          'python':platform.python_version(),'implementation':platform.python_implementation(),
          'libc':list(platform.libc_ver()),'logical_cpus':os.cpu_count(),
          'free_disk_mib':round(shutil.disk_usage(ROOT).free/2**20,1),'dependencies':{},
          'camera_devices':[],'hardware_claim':'Operator must identify target; architecture alone does not prove Feiteng board.'}
    try:
        info['memory_mib']=round(int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemTotal:')))/1024,1)
    except (OSError,StopIteration,ValueError): info['memory_mib']=None
    try:
        info['os']={k:v.strip('"') for k,v in (l.split('=',1) for l in Path('/etc/os-release').read_text().splitlines() if '=' in l) if k in ('ID','VERSION_ID','PRETTY_NAME')}
    except OSError: pass
    for name in PINS:
        try: info['dependencies'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: info['dependencies'][name]=None
    for device in sorted(Path('/dev').glob('video*')):
        info['camera_devices'].append({'device':device.name,'readable':os.access(device,os.R_OK),'writable':os.access(device,os.W_OK)})
    return info


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution-context',choices=['unknown','cloud','target-board'],default='unknown',help='Operator declaration only; not inferred from CPU')
    p.add_argument('--assets',type=Path,default=ROOT/'assets'); p.add_argument('--out',type=Path)
    p.add_argument('--mode',choices=['preflight','model-check','camera','recognize','enroll','field','historical','smoke','benchmark','all'],default='preflight')
    p.add_argument('--models',nargs='+',choices=['small160','large224'],default=['small160'])
    p.add_argument('--threads',type=int,nargs='+',default=[1,2,4]); p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--passes',type=int,default=1); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--camera',type=int,help='Explicit consent to open this camera; no images saved')
    p.add_argument('--image',type=Path,help='Actual local image for recognize mode; never collected')
    p.add_argument('--gallery',type=Path,help='Local generated gallery plus calibration receipt; recognize/field only')
    p.add_argument('--enrollment',type=Path,help='Optional field enrollment JSONL; contains private paths, kept local')
    p.add_argument('--field-queries',type=Path,help='Independent operator-verified known/unknown query JSONL')
    p.add_argument('--package-zip',type=Path,help='Original delivered ZIP for archive SHA256')
    a=p.parse_args(argv)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out=(a.out or ROOT/'results'/run_id).resolve(); out.mkdir(parents=True,exist_ok=False); out.chmod(0o700)
    private=out/'private'; private.mkdir(mode=0o700)
    report={'schema_version':3,'mode':a.mode,'run_id':run_id,'started_utc':datetime.now(timezone.utc).isoformat(),
            'source_commit':None,'package_manifest_sha256':None,'package_zip_sha256':None,'asset_manifest_sha256':None,
            'model_sha256':{'detector':None,'small160':None,'large224':None},'environment':{},'stages':[],'simulated':False,'recognition_validated':False,
            'hardware_validated':False,'operator_execution_context':a.execution_context,'scope':'Research pipeline diagnostics only; no production recognizer, no field accuracy claim.',
            'commands':{'requested':f'bash run.sh --mode {a.mode} --models {" ".join(a.models)} --threads {" ".join(map(str,a.threads))} --warmup {a.warmup} --passes {a.passes} --limit {a.limit} --assets <ASSET_DIR> --execution-context {a.execution_context}'},
            'privacy':'Only report.json, report.md and events.jsonl are collected. Raw logs/predictions/enrollment remain in private/; no images, private paths, environment variables or credentials collected.',
            'timing_policy':{'warmup':a.warmup,'passes':a.passes,'limit':a.limit,'first_frame':'measured separately; excluded from percentiles','samples':'raw measurements retained locally; summary counts are actual observations'}}
    for enabled, argument in [(a.image is not None, '--image <PRIVATE_IMAGE>'), (a.gallery is not None, '--gallery <LOCAL_GALLERY>'), (a.camera is not None, '--camera '+str(a.camera)), (a.enrollment is not None, '--enrollment <PRIVATE_ENROLLMENT_JSONL>'), (a.field_queries is not None, '--field-queries <PRIVATE_QUERY_JSONL>'), (a.package_zip is not None, '--package-zip <ORIGINAL_ZIP>')]:
        if enabled: report['commands']['requested'] += ' '+argument
    def stage(name,status,code,reason,**more):
        row={'stage':name,'status':status,'code':code,'reason':reason,**more}
        report['stages'].append(row)
        with (out/'events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(f'{name}: {status} {code} {reason}',flush=True)
        save(out/'report.json',report)
    def child(command,log,timeout=3600,progress=False):
        with (private/log).open('w',encoding='utf-8') as handle:
            try:
                if progress:
                    with (private/log).open(encoding='utf-8',errors='replace') as reader:
                        process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                            env=os.environ | {'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT),'PYTHONUNBUFFERED':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                        started=time.monotonic()
                        try:
                            while process.poll() is None:
                                text=reader.read()
                                if text: print(text,end='',flush=True)
                                if time.monotonic()-started>timeout:
                                    process.kill(); process.wait(); return 124
                                time.sleep(.2)
                            text=reader.read()
                            if text: print(text,end='',flush=True)
                            return process.returncode
                        finally:
                            if process.poll() is None:
                                process.kill(); process.wait()
                result=subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,timeout=timeout,
                    env=os.environ | {'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT),'PYTHONUNBUFFERED':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                return result.returncode
            except subprocess.TimeoutExpired: return 124
    try:
        if min(a.threads)<1 or a.warmup<0 or a.passes<1 or a.limit<0: raise ValueError('Invalid numeric options')
        report['environment']=environment()
        if (ROOT/'BUILD.json').exists(): report['source_commit']=json.loads((ROOT/'BUILD.json').read_text())['source_commit']
        if a.package_zip: report['package_zip_sha256']=digest(a.package_zip)
        try:
            verify_manifest(ROOT,'MANIFEST.json'); report['package_manifest_sha256']=digest(ROOT/'MANIFEST.json')
            stage('package_integrity','PASS','OK','All sealed package files match SHA256.')
            package_ok=True
        except Exception as exc:
            stage('package_integrity','FAIL','E_INTEGRITY','Package checksum missing/mismatch; inference disabled.',error_type=type(exc).__name__)
            package_ok=False
        env=report['environment']
        assessment=assess(env)
        report['platform_assessment']=assessment
        report['dependency_inventory']=inventory(env['dependencies'], PINS)
        report['recovery_command']='python3 -m experiments.devicekit.bootstrap'
        compatible=assessment['compatible']
        stage('preflight','PASS' if compatible else 'BLOCKED',
              'OK' if compatible else assessment['findings'][0]['code'],
              assessment['meaning'] if compatible else '; '.join(
                  f"{f['component']}: found {f['actual']}, require {f['required']}. {f['remediation']}"
                  for f in assessment['findings']))
        needs_model = a.mode in MODEL_MODES
        history = a.mode in HISTORY_MODES
        report['required_resources'] = plan(a.mode, a.models, bool(a.gallery))
        report['unrequested_work'] = 'Unrequested modes are SKIP, not missing-resource failures.'
        args_ok = True
        if len(set(a.models)) != len(a.models) or len(set(a.threads)) != len(a.threads):
            args_ok = False
        if a.mode in {'recognize','enroll','field'} and len(a.models) != 1:
            args_ok = False
        if (a.mode == 'camera' and a.camera is None) or (a.mode == 'recognize' and a.image is None):
            args_ok = False
        if a.mode in {'enroll','field'} and a.enrollment is None:
            args_ok = False
        if a.mode == 'field' and a.field_queries is None:
            args_ok = False
        if a.enrollment and a.mode not in {'enroll','field'} or a.field_queries and a.mode != 'field' or a.gallery and a.mode not in {'recognize','field'} or a.image and a.mode != 'recognize':
            args_ok = False
        if not args_ok:
            stage('arguments','FAIL','E_ARGS','Mode/arguments disagree; see the v2 mode table. No model inference requested until corrected.')
        config = None
        assets = a.assets.resolve()
        asset_ok = False
        if needs_model and args_ok:
            try:
                config, resource_hashes = verify_assets(assets, a.mode, a.models, bool(a.gallery))
                report['asset_manifest_sha256'] = digest(assets/'ASSET_MANIFEST.json')
                report['model_sha256']['detector'] = resource_hashes[config['detector']]
                for model in a.models:
                    report['model_sha256'][model] = resource_hashes[config['models'][model]['model']]
                report['verified_resource_count'] = len(resource_hashes)
                report['verified_resource_set_sha256'] = hashlib.sha256(json.dumps(resource_hashes,sort_keys=True).encode()).hexdigest()
                asset_ok = True
                stage('assets','PASS','OK','Selected source/license records, file hashes and preprocessing contracts verified. Technical verification is not legal clearance for public or commercial distribution.')
            except Exception as exc:
                with (private/'assets.log').open('w') as log:
                    log.write(type(exc).__name__+': '+str(exc)+'\n')
                stage('assets','BLOCKED','E_ASSETS','Required selected-mode assets are missing or inconsistent. See required_resources and private/assets.log.',error_type=type(exc).__name__,raw_log='private/assets.log')
        else:
            stage('assets','SKIP','NOT_REQUIRED','This mode does not consume model resources, or arguments need correction.')
        deps_ok = False
        if needs_model and args_ok and package_ok and compatible:
            imports = 'import numpy, PIL, onnxruntime; assert "CPUExecutionProvider" in onnxruntime.get_available_providers()'
            if history:
                imports += '; import psutil'
            dep_code=child([sys.executable,'-c',imports],'dependencies.log',60)
            deps_ok=dep_code==0
            stage('dependencies','PASS' if deps_ok else 'BLOCKED','OK' if deps_ok else 'E_DEPS',
                  'Selected inference runtime import/provider check.' if deps_ok else 'Required runtime import failed; see private/dependencies.log.',
                  required_modules=['numpy','PIL','onnxruntime']+(['psutil'] if history else []),
                  raw_log='private/dependencies.log',exit_code=dep_code,
                  error_summary=error_summary((private/'dependencies.log').read_text(encoding='utf-8',errors='replace')) if not deps_ok else {},
                  version_pin_match=all(env['dependencies'].get(k)==v for k,v in PINS.items() if history or k!='psutil'))
        else:
            stage('dependencies','SKIP','NOT_REQUIRED','No inference dependency imports needed; preflight inventory remains in environment.dependencies. Camera dependencies are checked separately.')
        inputs_ok = args_ok
        if a.mode in {'recognize','enroll','field'} and args_ok:
            try:
                from .enroll_field import validate_field
                if a.mode == 'recognize':
                    if not a.image.is_file(): raise ValueError('Input image unavailable')
                    report['input_image_sha256'] = digest(a.image)
                else:
                    erows=[json.loads(l) for l in a.enrollment.read_text().splitlines() if l.strip()]
                    validate_field(erows)
                    for r in erows:
                        if not (a.enrollment.parent/r['path']).is_file(): raise ValueError('Enrollment image unavailable')
                    report['enrollment_manifest_sha256']=digest(a.enrollment)
                    if a.mode == 'field':
                        qrows=[json.loads(l) for l in a.field_queries.read_text().splitlines() if l.strip()]
                        validate_field(qrows,query=True)
                        for key in ('session_id','event_id'):
                            if {r[key] for r in erows}&{r[key] for r in qrows}: raise ValueError('Field session/event overlap')
                        for r in qrows:
                            if not (a.field_queries.parent/r['path']).is_file(): raise ValueError('Query image unavailable')
                        report['query_manifest_sha256']=digest(a.field_queries)
                stage('local_inputs','PASS','OK','Requested local inputs exist and declared field identities/sessions validated; image decode is still required.')
            except Exception as exc:
                inputs_ok=False
                with (private/'inputs.log').open('w') as log: log.write(type(exc).__name__+': '+str(exc)+'\n')
                stage('local_inputs','BLOCKED','E_INPUT','Requested local image/field data missing or invalid; no historical images required.',raw_log='private/inputs.log',error_type=type(exc).__name__)
        gallery_path = None
        gallery_ok = True
        if a.gallery and args_ok:
            if asset_ok:
                try:
                    details=verify_local_gallery(a.gallery.resolve(),assets,config,a.models[0])
                    if a.mode == 'field':
                        receipt=json.loads(Path(str(a.gallery.resolve())+'.provenance.json').read_text())
                        if receipt['enrollment_manifest_sha256'] != digest(a.enrollment): raise ValueError('Enrollment does not match local gallery receipt')
                    gallery_path=a.gallery.resolve()
                    stage('local_gallery','PASS','OK','Generated gallery and calibration receipt match selected model/configuration.',**details)
                except Exception as exc:
                    gallery_ok=False
                    with (private/'gallery.log').open('w') as log: log.write(type(exc).__name__+': '+str(exc)+'\n')
                    stage('local_gallery','BLOCKED','E_GALLERY','Local gallery or threshold provenance cannot be verified; unknown threshold is never accepted.',raw_log='private/gallery.log',error_type=type(exc).__name__)
            else:
                gallery_ok=False
                stage('local_gallery','BLOCKED','E_PREREQUISITE','Selected model assets required to verify gallery binding.')
        ready=package_ok and compatible and deps_ok and asset_ok and inputs_ok and gallery_ok and args_ok
        loaded={}
        if needs_model and ready:
            for model in a.models:
                load_gallery = gallery_path if a.mode == 'recognize' and a.gallery else assets/config['models'][model]['gallery'] if history or a.mode == 'recognize' else None
                command=[sys.executable,'-m','experiments.devicekit.validate_model',str(assets),model]
                if load_gallery is not None: command.append(str(load_gallery))
                command += ['--threads',str(a.threads[0])]
                rc=child(command,f'load-{model}.log',180)
                loaded[model]=rc==0
                evidence={}
                if rc == 0:
                    evidence=json.loads((private/f'load-{model}.log').read_text(encoding='utf-8').splitlines()[-1])
                stage('model_load:'+model,'PASS' if rc==0 else 'FAIL','OK' if rc==0 else 'E_MODEL','Actual CPU ONNX graph interfaces checked; no image inference in model-check.',raw_log=f'private/load-{model}.log',exit_code=rc,evidence=evidence,
                      error_summary=error_summary((private/f'load-{model}.log').read_text(encoding='utf-8',errors='replace')) if rc else {})
        elif needs_model:
            stage('model_load','BLOCKED','E_PREREQUISITE','No model loaded; selected-mode prerequisites missing.')
        else:
            stage('model_load','SKIP','NOT_REQUESTED','Pure environment/camera diagnostics do not load models.')
        if history:
            if ready:
                for model in a.models:
                    if not loaded.get(model):
                        stage('historical:'+model,'BLOCKED','E_MODEL','Selected model failed to load.')
                        continue
                    for threads in a.threads:
                        name=f'{model}-t{threads}'; target=private/name; target.mkdir()
                        limit=4 if a.mode=='smoke' else a.limit
                        command=[sys.executable,'-m','experiments.board.run','--bundle',str(assets),'--out',str(target),'--worker',model,'--threads',str(threads),'--warmup',str(a.warmup),'--passes',str(a.passes),'--limit',str(limit)]
                        print(f'Running {name}: {a.passes} pass(es), {limit or 149} queries/pass; please wait.',flush=True)
                        rc=child(command,name+'.log',timeout=14400,progress=True)
                        summary_path=target/'summary.json'
                        if rc or not summary_path.exists():
                            stage('historical:'+name,'FAIL','E_INFERENCE','Historical replay failed; no inference scores invented.',exit_code=rc,raw_log='private/'+name+'.log',
                                  error_summary=error_summary((private/(name+'.log')).read_text(encoding='utf-8',errors='replace')))
                            continue
                        summary=json.loads(summary_path.read_text())
                        stage('image_smoke:'+name,'PASS' if summary['latency_ms']['identity_and_match_ms']['count'] else 'BLOCKED','OK' if summary['latency_ms']['identity_and_match_ms']['count'] else 'E_NO_IDENTITY_SAMPLE','Actual replay; identity smoke requires a usable single-cat branch.',raw_log='private/'+name+'.log',raw_measurements='private/'+name+'/measurements.jsonl',identity_branch_samples=summary['latency_ms']['identity_and_match_ms']['count'])
                        stage('known_unknown:'+name,'PASS','OK','Historical measurement, not field accuracy acceptance.',metrics=summary['metrics'],threshold=summary['threshold'],limited_run=summary['limited_run'])
                        stage('benchmark:'+name,'PASS','OK','Observed timings; no board speed extrapolation.',**{k:summary[k] for k in ('latency_ms','model_load_ms','first_frame_ms','peak_rss_mib','providers','unique_queries','passes','warmup','timing_scope','limited_run','pass_summaries','telemetry') if k in summary})
            else:
                stage('historical','BLOCKED','E_PREREQUISITE','Historical replay needs frozen protocol, gallery evidence and all 149 images, plus selected models.')
        else:
            stage('historical','SKIP','NOT_REQUESTED','No frozen historical protocol, gallery or 149-image dataset required.')
        if a.mode == 'recognize':
            if ready and loaded.get(a.models[0]):
                gp=gallery_path or assets/config['models'][a.models[0]]['gallery']
                rc=child([sys.executable,'-m','experiments.devicekit.recognize_image',str(assets),a.models[0],str(gp),str(a.image.resolve()),str(private)],'recognize.log')
                summary=json.loads((private/'image-summary.json').read_text()) if (private/'image-summary.json').exists() else {}
                stage('image_recognition','PASS' if rc==0 else 'FAIL','OK' if rc==0 else 'E_INFERENCE','Actual one-image pipeline observation; predictions stay private, no accuracy claim.',raw_log='private/recognize.log',summary=summary,exit_code=rc)
            else:
                stage('image_recognition','BLOCKED','E_PREREQUISITE','Needs selected detector/identity model, calibrated matching gallery, runtime and actual image.')
        else:
            stage('image_recognition','SKIP','NOT_REQUESTED','One-image recognition not requested.')
        if a.mode in {'enroll','field'}:
            enrolled=ready and loaded.get(a.models[0],False)
            if enrolled and not a.gallery:
                gallery_path=private/'field-gallery.json'
                rc=child([sys.executable,'-m','experiments.devicekit.enroll_field',str(a.enrollment.resolve()),str(assets),a.models[0],str(gallery_path)],'enrollment.log')
                enrolled=rc==0
                stage('enrollment','PASS' if enrolled else 'FAIL','OK' if enrolled else 'E_ENROLL','Actual field enrollment and calibration score replay; no historical gallery/query dependency.',raw_log='private/enrollment.log',exit_code=rc)
            elif enrolled:
                stage('enrollment','SKIP','EXISTING_GALLERY','Hash-bound local gallery reused; no rebuilding requested.')
            else:
                stage('enrollment','BLOCKED','E_PREREQUISITE','Needs detector + selected identity model + real registration/calibration inputs, not an old gallery.')
            if a.mode == 'field':
                if enrolled:
                    rc=child([sys.executable,'-m','experiments.devicekit.field_queries',str(a.enrollment.resolve()),str(a.field_queries.resolve()),str(assets),a.models[0],str(gallery_path),str(private)],'field-query.log')
                    summary=json.loads((private/'field-summary.json').read_text()) if (private/'field-summary.json').exists() else {}
                    stage('field_known_unknown','PASS' if rc==0 else 'FAIL','OK' if rc==0 else 'E_FIELD','Independent local field query diagnostic; threshold/calibration provenance retained.',raw_log='private/field-query.log',summary=summary,exit_code=rc)
                else:
                    stage('field_known_unknown','BLOCKED','E_PREREQUISITE','Needs usable field gallery and independent actual queries.')
        else:
            stage('enrollment','SKIP','NOT_REQUESTED','No field enrollment requested.')
        if a.camera is not None and package_ok and compatible and args_ok:
            rc=child([sys.executable,'-m','experiments.devicekit.camera',str(a.camera)],'camera.log',30)
            stage('camera','PASS' if rc==0 else 'BLOCKED','OK' if rc==0 else 'E_CAMERA','One-frame capture diagnostic; frame discarded, no identity assertion.',raw_log='private/camera.log',exit_code=rc)
        elif a.camera is not None: stage('camera','BLOCKED','E_PREREQUISITE','Package/platform validation failed; no camera opened.')
        else: stage('camera','SKIP','NOT_REQUESTED','No camera opened. Use --camera INDEX only with site consent.')
    except Exception as exc:
        stage('supervisor','FAIL','E_INTERNAL','Supervisor failed; report retained. Check documented arguments and input JSON.',error_type=type(exc).__name__)
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        statuses={s['status'] for s in report['stages']}
        report['overall']='FAIL' if 'FAIL' in statuses else 'BLOCKED' if 'BLOCKED' in statuses else 'PASS'
        report['overall_meaning']='PASS means requested diagnostics completed; never production/field recognition acceptance.'
        save(out/'report.json',report)
        lines=['# CatFace 设备测试报告',f"run_id: {run_id}",f"总状态: **{report['overall']}**",'',report['scope'],'',f"源码: `{report['source_commit']}`",f"包清单 SHA256: `{report['package_manifest_sha256']}`",f"ZIP SHA256: `{report['package_zip_sha256']}`",f"模型 SHA256: `{json.dumps(report['model_sha256'])}`",'', '未验证飞腾派硬件；环境为本次执行机器。无模拟识别成绩。PASS 仅代表该项诊断完成。','','| 阶段 | 状态 | 代码 | 说明 |','|---|---|---|---|']
        for s in report['stages']: lines.append('| '+ ' | '.join(str(s.get(k,'')) for k in ('stage','status','code','reason'))+' |')
        lines+=['','软件/设备环境：','```json',json.dumps(report['environment'],ensure_ascii=False,indent=2),'```',
                '', '运行环境诊断：', '```json', json.dumps(report.get('platform_assessment',{}),ensure_ascii=False,indent=2),'```',
                '', '依赖版本清单（不代表成功导入）：', '```json',json.dumps(report.get('dependency_inventory',[]),ensure_ascii=False,indent=2),'```',
                '', '离线修复入口：`python3 -m experiments.devicekit.bootstrap`；只在匹配的 Linux ARM64/glibc 设备安装包内运行时，不替换系统 Python。',
                '', '详细计时分布、样本数及预热规则见 report.json。原始日志保留 private/，默认回传不含此目录。','复跑：`'+report['commands']['requested']+'`']
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('report.json','report.md','events.jsonl'):
                if (out/name).exists(): z.write(out/name,name)
        archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
        print('Report:',out); print('Return only:',archive)
    return {'PASS':0,'BLOCKED':2,'FAIL':1}[report['overall']]

if __name__=='__main__': raise SystemExit(main())
