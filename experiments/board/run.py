"""Offline full-frame replay and board measurements; no training or calibration."""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sensors():
    """Unavailable sysfs sensors remain absent, rather than being reported as zero."""
    result = {'temperature_c': {}, 'cpu_frequency_khz': {}}
    for pattern, key, scale in [('/sys/class/thermal/thermal_zone*/temp', 'temperature_c', 1000),
                                ('/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq', 'cpu_frequency_khz', 1)]:
        import glob
        for name in glob.glob(pattern):
            try:
                result[key][Path(name).parent.name] = float(Path(name).read_text().strip()) / scale
            except (OSError, ValueError):
                pass
    return result


def device_info():
    import numpy
    import PIL
    import onnxruntime
    import psutil
    result = {'platform': platform.platform(), 'machine': platform.machine(),
              'python': sys.version, 'logical_cpus': os.cpu_count(),
              'ram_mib': psutil.virtual_memory().total / 2**20,
              'versions': {'numpy': numpy.__version__, 'Pillow': PIL.__version__,
                           'onnxruntime': onnxruntime.__version__, 'psutil': psutil.__version__},
              'sensors': sensors()}
    for name in ('/proc/cpuinfo', '/etc/os-release', '/proc/device-tree/model'):
        try:
            result[name] = Path(name).read_text().replace('\x00', '').strip()
        except OSError:
            pass
    return result


def identity_metrics(rows):
    known = [r for r in rows if r['role'] == 'known']
    unknown = [r for r in rows if r['role'] == 'unknown']
    correct = sum(r['status'] == 'known' and r.get('cat_id') == r['true_cat_id'] for r in known)
    wrong = sum(r['status'] == 'known' and r.get('cat_id') != r['true_cat_id'] for r in known)
    false = sum(r['status'] == 'known' for r in unknown)
    return {'known_queries': len(known), 'unknown_queries': len(unknown),
            'known_correct_accepts': correct, 'known_wrong_accepts': wrong,
            'unknown_false_accepts': false,
            'dir': correct / len(known) if known else None,
            'fpir': false / len(unknown) if unknown else None,
            'known_status_counts': dict(Counter(r['status'] for r in known)),
            'unknown_status_counts': dict(Counter(r['status'] for r in unknown)),
            'scope': 'First measured pass only. All query failures stay in denominators. no_cat/unusable/error are not successful unknown rejections.'}


def telemetry_summary(rows):
    result = {}
    for key in ('temperature_c', 'cpu_frequency_khz'):
        names = sorted({name for row in rows for name in row.get(key, {})})
        result[key] = {}
        for name in names:
            values = [row[key][name] for row in rows if name in row.get(key, {})]
            result[key][name] = dict(first=values[0], last=values[-1], min=min(values), max=max(values))
    rss = [row['rss_mib'] for row in rows]
    result['rss_mib'] = dict(first=rss[0], last=rss[-1], max=max(rss)) if rss else None
    return result


def worker(args, config):
    import numpy as np
    import psutil
    from experiments.detection.onnx import Detector
    from experiments.detection.replay import replay, calibration_scope
    from experiments.identity.runtime import Encoder, read_image
    from experiments.identity.deploy import read_gallery

    rows = [json.loads(line) for line in (args.bundle / config['manifest']).read_text(encoding='utf-8').splitlines() if line.strip()]
    if args.limit:
        # Interleave known/unknown for a small, useful smoke test.
        known = [r for r in rows if r['role'] == 'known']
        unknown = [r for r in rows if r['role'] == 'unknown']
        rows = [r for i in range(max(len(known), len(unknown))) for group in (known, unknown) for r in group[i:i+1]][:args.limit]
    if not rows:
        raise ValueError('query manifest is empty')
    model = config['models'][args.worker]
    started = time.perf_counter()
    detector = Detector(args.bundle / config['detector'], args.threads[0])
    encoder = Encoder(args.bundle / model['model'], args.threads[0])
    gallery = read_gallery(args.bundle / model['gallery'], encoder)
    calibration_scope(gallery, detector)
    load_ms = (time.perf_counter() - started) * 1000
    for row in rows:
        if row['role'] not in ('known', 'unknown'):
            raise ValueError(f'unsupported query role: {row["role"]}')
        if (row['cat_id'] in gallery['cat_ids']) != (row['role'] == 'known'):
            raise ValueError('query role does not match frozen gallery')
        if not (args.bundle / row['path']).is_file():
            raise FileNotFoundError(row['path'])

    def run_one(row):
        begin = time.perf_counter()
        image = read_image(args.bundle / row['path'])
        decoded = time.perf_counter()
        try:
            result = replay(image, detector, encoder, gallery)
        finally:
            image.close()
        result.update(decode_ms=(decoded - begin) * 1000,
                      total_ms=(time.perf_counter() - begin) * 1000)
        return result

    def observe(row):
        try:
            return run_one(row)
        except Exception as exc:
            return {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}

    first = observe(rows[0])
    warmup_errors = 0
    for i in range(args.warmup):
        warmup_errors += observe(rows[i % len(rows)])['status'] == 'error'
    stages = {k: [] for k in ('decode_ms', 'detect_ms', 'identity_and_match_ms', 'total_ms')}
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    begin = time.perf_counter()
    predictions = []
    pass_summaries = []
    telemetry_rows = []
    errors = 0
    with (args.out / 'measurements.jsonl').open('w', encoding='utf-8') as measured, (args.out / 'telemetry.jsonl').open('w', encoding='utf-8') as telemetry:
        for repeat in range(args.passes):
            pass_rows = []
            for index, row in enumerate(rows):
                result = observe(row)
                errors += result['status'] == 'error'
                result.update(sample_id=row['sample_id'], path=row['path'], true_cat_id=row['cat_id'], role=row['role'], repeat=repeat)
                pass_rows.append(result)
                measured.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                measured.flush()
                if repeat == 0:
                    predictions.append(result)
                for key in stages:
                    if key in result and (key != 'identity_and_match_ms' or result['status'] in ('known', 'unknown')):
                        stages[key].append(result[key])
                reading = {'elapsed_s': time.perf_counter() - begin,
                           'rss_mib': proc.memory_info().rss / 2**20, **sensors()}
                telemetry_rows.append(reading)
                telemetry.write(json.dumps(reading) + '\n')
                if (index + 1) % 10 == 0 or index + 1 == len(rows):
                    print(f'{args.worker} t{args.threads[0]} pass {repeat+1}/{args.passes}: {index+1}/{len(rows)}', flush=True)
            totals = [r['total_ms'] for r in pass_rows if 'total_ms' in r]
            pass_summaries.append({'repeat': repeat, 'queries': len(pass_rows),
                'errors': sum(r['status'] == 'error' for r in pass_rows),
                'total_p50_ms': float(np.percentile(totals, 50)) if totals else None,
                'total_p95_ms': float(np.percentile(totals, 95)) if totals else None})
    elapsed = time.perf_counter() - begin
    cpu_after = proc.cpu_times()
    memory = proc.memory_info()
    if platform.system() == 'Windows':
        peak = memory.peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
    latency = {k: {'count': len(v), 'p50': float(np.percentile(v, 50)) if v else None,
                   'p95': float(np.percentile(v, 95)) if v else None,
                   'mean': float(np.mean(v)) if v else None} for k, v in stages.items()}
    summary = {'model': args.worker, 'threads': args.threads[0], 'unique_queries': len(rows),
               'passes': args.passes, 'warmup': args.warmup, 'errors': errors,
               'threshold': gallery['threshold'], 'model_sha256': gallery['model_sha256'],
               'crop_pipeline': gallery['crop_pipeline'],
               'providers': {'detector': detector.session.get_providers(), 'identity': encoder.session.get_providers()},
               'model_load_ms': load_ms, 'first_frame_ms': first.get('total_ms'),
               'first_frame_status': first['status'], 'warmup_errors': warmup_errors,
               'pass_summaries': pass_summaries, 'telemetry': telemetry_summary(telemetry_rows),
               'latency_ms': latency, 'peak_rss_mib': peak / 2**20,
               'elapsed_s': elapsed, 'completed_fps': sum(len(v) for k, v in stages.items() if k == 'total_ms') / elapsed,
               'cpu_percent_one_core': 100 * (cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system) / elapsed,
               'metrics': identity_metrics(predictions), 'diagnostic_only': True,
               'timing_scope': 'total includes file read, EXIF/RGB decode, detection/crop, identity/match; excludes model load and output writes. Repeated reads may use OS file cache. Identity timing conditional on usable single crop. Errors have no latency; see errors count.',
               'dataset': config['dataset'], 'limited_run': bool(args.limit)}
    save(args.out / 'summary.json', summary)
    return 1 if errors or warmup_errors or first['status'] == 'error' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path)
    parser.add_argument('--models', nargs='+', default=['small160', 'large224'])
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--passes', type=int, default=1, help='repeat measurements; accuracy uses first pass only')
    parser.add_argument('--limit', type=int, default=0, help='smoke test query limit; 0 runs all')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads) < 1 or args.warmup < 0 or args.passes < 1 or args.limit < 0:
        parser.error('threads/passes must be positive; warmup/limit non-negative')
    args.bundle = args.bundle.resolve()
    config = json.loads((args.bundle / 'bundle.json').read_text(encoding='utf-8'))
    if args.worker:
        return worker(args, config)
    if len(set(args.models)) != len(args.models) or len(set(args.threads)) != len(args.threads):
        parser.error('models/threads must not contain duplicates')
    if set(args.models) - config['models'].keys():
        parser.error('unknown model name')
    out = (args.out or args.bundle / 'results' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'device.json', device_info())
    save(out / 'run.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
                           'arguments': vars(args) | {'bundle': str(args.bundle), 'out': str(out)}, 'bundle': config})
    summaries = []
    failed = []
    for model in args.models:
        for threads in args.threads:
            target = out / f'{model}-t{threads}'
            target.mkdir()
            command = [sys.executable, '-m', 'experiments.board.run', '--bundle', str(args.bundle),
                       '--out', str(target), '--worker', model, '--threads', str(threads),
                       '--warmup', str(args.warmup), '--passes', str(args.passes), '--limit', str(args.limit)]
            with (target / 'run.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           env=os.environ | {'PYTHONUNBUFFERED': '1'})
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                code = process.wait()
            if code:
                failed.append(target.name)
            if (target / 'summary.json').exists():
                summaries.append(json.loads((target / 'summary.json').read_text(encoding='utf-8')))
    save(out / 'summary.json', {'complete': not failed, 'failed': failed, 'results': summaries})
    with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'threads', 'queries', 'DIR', 'FPIR', 'total_p50_ms', 'total_p95_ms', 'peak_rss_mib', 'errors'])
        for item in summaries:
            writer.writerow([item['model'], item['threads'], item['unique_queries'], item['metrics']['dir'], item['metrics']['fpir'],
                             item['latency_ms']['total_ms']['p50'], item['latency_ms']['total_ms']['p95'], item['peak_rss_mib'], item['errors']])
    archive = out.with_name(out.name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname=out.name)
    print(f'Results: {out}\nCopy back: {archive}\nStatus: {"FAILED (see run.log)" if failed else "OK"}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
