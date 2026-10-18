"""One board visit: offline setup, full replay matrix, optional field/camera, one return ZIP."""
import argparse
import csv
import json
import os
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .build import digest
from .platform_check import error_summary
from .resources import verify_manifest

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def collect_replay(private):
    """Export anonymous observations and summaries, never image paths or raw logs."""
    summaries, observations, comparisons = [], [], []
    reference, samples, cats = {}, {}, {}

    def anonymous(mapping, value, prefix):
        if value is None:
            return None
        if value not in mapping:
            mapping[value] = prefix + str(len(mapping) + 1).zfill(3)
        return mapping[value]

    for folder in sorted(private.glob('*-t*')):
        if not (folder / 'summary.json').is_file():
            continue
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        summaries.append(summary)
        for line in (folder / 'measurements.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            sample = anonymous(samples, row['sample_id'], 'sample-')
            key = (summary['model'], sample)
            decision = (row['status'], row.get('cat_id'))
            if key not in reference:
                reference[key] = (decision, row.get('score'), folder.name)
            baseline, score, _ = reference[key]
            comparisons.append(dict(model=summary['model'], configuration=folder.name,
                decision_changed=decision != baseline,
                score_delta=abs(row['score'] - score) if row.get('score') is not None and score is not None else None))
            exported = {k: row[k] for k in ('role', 'status', 'repeat', 'score', 'decode_ms',
                'detect_ms', 'identity_and_match_ms', 'total_ms') if k in row}
            exported.update(configuration=folder.name, sample_id=sample,
                true_cat_id=anonymous(cats, row['true_cat_id'], 'cat-'),
                cat_id=anonymous(cats, row.get('cat_id'), 'cat-'),
                top_candidate_id=anonymous(cats, row.get('top_candidate_id'), 'cat-'))
            if 'error' in row:
                exported['error_summary'] = error_summary(row['error'])
            observations.append(exported)
    consistency = []
    for model in sorted({r['model'] for r in summaries}):
        rows = [r for r in comparisons if r['model'] == model]
        deltas = [r['score_delta'] for r in rows if r['score_delta'] is not None]
        consistency.append(dict(model=model, observations=len(rows),
            decision_changes=sum(r['decision_changed'] for r in rows),
            max_score_delta=max(deltas) if deltas else None,
            reference='First configuration and first measured pass per image; same model only'))
    return summaries, observations, consistency


def render_summary(report, summaries):
    lines = ['# CatFace 一次完整板端测试', '', '**状态：' + report['overall'] + '**', '',
        '固定公开图 development 诊断；不代表现场准确率或生产验收。', '',
        '| 模型 | 线程 | 查询数 | 正确接受/已知 | 误接受/陌生 | P50 ms | P95 ms | 峰值 MiB | 错误 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        m, latency = r['metrics'], r['latency_ms']['total_ms']
        number = lambda x: '—' if x is None else f'{x:.2f}'
        lines.append('| ' + ' | '.join(map(str, [r['model'], r['threads'], r['unique_queries'],
            f"{m['known_correct_accepts']}/{m['known_queries']}",
            f"{m['unknown_false_accepts']}/{m['unknown_queries']}",
            number(latency['p50']), number(latency['p95']), number(r['peak_rss_mib']), r['errors']])) + ' |')
    lines += ['', '每组首次推理、分阶段耗时、分轮变化、温度/频率和一致性见 summary.json。',
        '准确率仅用第一轮；重复轮次增加计时观测，不增加独立身份样本。', '', '执行阶段：', '']
    for stage in report['stages']:
        lines.append('- ' + stage['stage'] + ': ' + stage['status'])
    if report.get('error_type'):
        lines += ['', '运行异常：' + report['error_type'] + '；详细日志留在本机 private/。']
    lines += ['', '回传本次结果 ZIP 即可；不需要回传照片、图库或 private/。']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--models', nargs='+', choices=['small160', 'large224'], default=['small160', 'large224'])
    p.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--passes', type=int, default=3)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--limit', type=int, default=0, help='0: all 149 images; nonzero marks a limited diagnostic')
    p.add_argument('--execution-context', choices=['unknown', 'cloud', 'target-board'], default='target-board')
    p.add_argument('--enrollment', type=Path)
    p.add_argument('--field-queries', type=Path)
    p.add_argument('--camera', type=int, help='Explicitly request one-frame camera capture; requires local cv2')
    a = p.parse_args(argv)
    if min(a.threads) < 1 or a.passes < 1 or a.warmup < 0 or a.limit < 0:
        p.error('Invalid threads/passes/warmup/limit')
    if len(set(a.models)) != len(a.models) or len(set(a.threads)) != len(a.threads):
        p.error('Duplicate models/threads')
    if bool(a.enrollment) != bool(a.field_queries):
        p.error('--enrollment and --field-queries must be provided together')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-suite-' + uuid.uuid4().hex[:8]
    out = (a.out or ROOT / 'results' / run_id).resolve()
    out.mkdir(parents=True, exist_ok=False)
    private = out / 'private'
    private.mkdir(mode=0o700)
    report = dict(schema_version=1, mode='suite', run_id=run_id, stages=[],
        execution_context=a.execution_context, limited_run=bool(a.limit),
        requested_configurations=len(a.models) * len(a.threads),
        started_utc=datetime.now(timezone.utc).isoformat(),
        scope='Frozen development images, publisher identities; not independent field or production acceptance')
    summaries, observations, consistency = [], [], []
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'src') + os.pathsep + str(ROOT), PYTHONUNBUFFERED='1',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def stage(name, status, **details):
        report['stages'].append(dict(stage=name, status=status, **details))
        save(out / 'report.json', report)
        print(name + ': ' + status, flush=True)

    def execute(command, log):
        with (private / log).open('w', encoding='utf-8') as handle:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
                for line in process.stdout:
                    if not line.startswith('Return only:'):
                        print(line, end='', flush=True)
                    handle.write(line)
                return process.wait()

    def diagnostic(name, python, options):
        target = private / name
        code = execute([str(python), '-m', 'experiments.devicekit.run', '--out', str(target),
            '--execution-context', a.execution_context] + options, name + '.log')
        path = target / 'report.json'
        detail = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        stage(name, detail['overall'] if detail else 'FAIL', exit_code=code)
        if detail:
            save(out / (name + '.json'), detail)
        return detail

    try:
        verify_manifest(ROOT, 'MANIFEST.json')
        report['package_manifest_sha256'] = digest(ROOT / 'MANIFEST.json')
        report['source_commit'] = json.loads((ROOT / 'BUILD.json').read_text(encoding='utf-8'))['source_commit']
        stage('package_integrity', 'PASS')
        python = Path(sys.executable)
        probe = subprocess.run([str(python), '-c',
            'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12); import numpy,PIL,onnxruntime,psutil; '
            'assert "CPUExecutionProvider" in onnxruntime.get_available_providers()'],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        if probe.returncode:
            previous_boot_reports=set((ROOT / 'results').glob('*-bootstrap-*/report.json'))
            rc = execute([str(python), '-m', 'experiments.devicekit.bootstrap'], 'bootstrap.log')
            # Include the bootstrap JSON even if installation failed.
            boot = sorted(set((ROOT / 'results').glob('*-bootstrap-*/report.json')) - previous_boot_reports,
                          key=lambda f: f.stat().st_mtime)
            if boot:
                save(out / 'bootstrap.json', json.loads(boot[-1].read_text(encoding='utf-8')))
            stage('runtime_setup', 'PASS' if rc == 0 else 'BLOCKED', exit_code=rc)
            if rc:
                return 2
            python = ROOT / '.runtime/python/bin/python3'
        else:
            stage('runtime_setup', 'PASS', reused_existing_runtime=True)
        diagnostic('historical', python, ['--mode', 'historical', '--models'] + a.models +
            ['--threads'] + list(map(str, a.threads)) + ['--passes', str(a.passes),
            '--warmup', str(a.warmup), '--limit', str(a.limit)])
        summaries, observations, consistency = collect_replay(private / 'historical/private')
        complete = len(summaries) == report['requested_configurations']
        stage('replay_matrix', 'PASS' if complete and all(r['errors'] == 0 and not r.get('warmup_errors') and
            r.get('first_frame_status') != 'error' for r in summaries) else 'FAIL',
            completed_configurations=len(summaries), requested_configurations=report['requested_configurations'])
        changed = sum(r['decision_changes'] for r in consistency)
        stage('repeat_thread_consistency', 'PASS' if complete and not changed else 'BLOCKED',
            decision_changes=changed)
        # Optional inputs never prevent the bundled replay from completing.
        if a.enrollment:
            for model in a.models:
                diagnostic('field-' + model, python, ['--mode', 'field', '--models', model,
                    '--enrollment', str(a.enrollment.resolve()), '--field-queries', str(a.field_queries.resolve())])
        else:
            stage('field', 'SKIP', reason='No operator-verified field enrollment/queries supplied')
        if a.camera is not None:
            diagnostic('camera', python, ['--mode', 'camera', '--camera', str(a.camera)])
        else:
            stage('camera', 'SKIP', reason='Not requested')
    except KeyboardInterrupt:
        stage('interrupted', 'FAIL', reason='Interrupted; partial results retained')
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        (private / 'failure.log').write_text(type(exc).__name__ + ': ' + str(exc), encoding='utf-8')
        stage('suite', 'FAIL', error_summary=error_summary(type(exc).__name__ + ': ' + str(exc)))
    finally:
        if not summaries:
            summaries, observations, consistency = collect_replay(private / 'historical/private')
        statuses = {r['status'] for r in report['stages']}
        report['overall'] = 'FAIL' if 'FAIL' in statuses else 'BLOCKED' if 'BLOCKED' in statuses else 'PASS'
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(out / 'report.json', report)
        fastest = {model: min((r for r in summaries if r['model'] == model and not r['errors'] and
            r['latency_ms']['total_ms']['p95'] is not None),
            key=lambda r: r['latency_ms']['total_ms']['p95'], default=None) for model in a.models}
        save(out / 'summary.json', dict(results=summaries, consistency=consistency,
            fastest_observed_threads={k: v['threads'] if v else None for k, v in fastest.items()},
            note='Fastest observed P95 per model; no production model selection or field accuracy claim'))
        (out / 'observations.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
            for r in observations), encoding='utf-8')
        with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['model', 'threads', 'queries', 'passes', 'DIR', 'FPIR', 'P50_ms', 'P95_ms',
                'first_frame_ms', 'peak_rss_mib', 'errors'])
            for r in summaries:
                writer.writerow([r['model'], r['threads'], r['unique_queries'], r['passes'], r['metrics']['dir'],
                    r['metrics']['fpir'], r['latency_ms']['total_ms']['p50'], r['latency_ms']['total_ms']['p95'],
                    r['first_frame_ms'], r['peak_rss_mib'], r['errors']])
        (out / 'report.md').write_text(render_summary(report, summaries), encoding='utf-8')
        archive = out.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                if f.is_file():
                    z.write(f, f.name)
        archive.with_suffix('.zip.sha256').write_text(digest(archive) + '  ' + archive.name + '\n', encoding='utf-8')
        print('\nReturn only: ' + str(archive) + '\nStatus: ' + report['overall'], flush=True)
    return {'PASS': 0, 'BLOCKED': 2, 'FAIL': 1}[report['overall']]


if __name__ == '__main__':
    raise SystemExit(main())
