"""Prepare a relocatable full-frame CPU test bundle from local research assets."""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments/runs/feiteng-offline')
    parser.add_argument('--python-version', default='3.11', help='target CPython minor version')
    parser.add_argument('--platform', default='manylinux_2_27_aarch64', help='target pip wheel platform (glibc 2.27+)')
    parser.add_argument('--skip-wheels', action='store_true', help='for targets with dependencies already installed')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) or out.with_suffix('.tar.gz').exists():
        parser.error('output directory/archive must be new; choose another --out')

    def copy(source, destination):
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, target)

    for name in ('__init__.py', 'runtime.py', 'deploy.py'):
        copy(f'experiments/identity/{name}', f'experiments/identity/{name}')
    for name in ('onnx.py', 'containment.py', 'replay.py'):
        copy(f'experiments/detection/{name}', f'experiments/detection/{name}')
    copy('experiments/board/run.py', 'experiments/board/run.py')
    for name in ('run.sh', 'requirements.txt', 'README.md'):
        copy(f'experiments/board/{name}', name)
    detector = 'experiments/runs/detection-cpu/onnx'
    for name in ('model.onnx', 'model.onnx.json'):
        copy(f'{detector}/{name}', f'detector/{name}')
    models = {}
    for name, folder in [('small160', 'edge160/baseline'), ('large224', 'backbone-study/large224')]:
        for file in ('model.onnx', 'model.json'):
            copy(f'experiments/runs/{folder}/{file}', f'models/{name}/{file}')
        copy(f'experiments/runs/crop-study/{name}/fixed/detected.gallery.json', f'models/{name}/gallery.json')
        models[name] = {'model': f'models/{name}/model.onnx', 'gallery': f'models/{name}/gallery.json'}

    protocol = json.loads((ROOT / 'experiments/runs/edge160/protocol.json').read_text(encoding='utf-8'))
    clean = {row['path']: row for row in map(json.loads, (ROOT / 'data/clean.identity.jsonl').read_text(encoding='utf-8').splitlines())}
    queries = []
    # Fixed development queries only: neither enrollment nor threshold-calibration images.
    for role in ('known', 'unknown'):
        for row in protocol['development'][role]:
            original = clean[row['path']]['original_path']
            destination = f"images/{len(queries):04d}{Path(original).suffix}"
            copy(f'data/{original}', destination)
            queries.append({'sample_id': row['sample_id'], 'path': destination,
                            'source_path': original, 'cat_id': row['cat_id'], 'role': role})
    (out / 'queries.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in queries), encoding='utf-8')
    config = {'detector': 'detector/model.onnx', 'models': models, 'manifest': 'queries.jsonl',
              'wheel_python': args.python_version, 'wheel_platform': args.platform,
              'dataset': 'edge160 fixed development full-frame queries; publisher labels; no visit/session truth',
              'diagnostic_only': True}
    (out / 'bundle.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    if not args.skip_wheels:
        platforms = ['--platform', args.platform]
        # pip cross-download needs explicit older compatible tags as well.
        architecture = 'aarch64' if args.platform.endswith('_aarch64') else 'x86_64'
        if args.platform.startswith('manylinux_2_27_'):
            platforms += ['--platform', f'manylinux2014_{architecture}',
                          '--platform', f'manylinux_2_17_{architecture}']
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                        *platforms, '--python-version', args.python_version,
                        '--implementation', 'cp', '--dest', str(out / 'wheels'),
                        '-r', str(out / 'requirements.txt')], check=True)
    with tarfile.open(out.with_suffix('.tar.gz'), 'w:gz', compresslevel=1) as tar:
        tar.add(out, arcname=out.name)
    print(f'Prepared {len(queries)} original images, two models: {out.with_suffix(".tar.gz")}')


if __name__ == '__main__':
    main()
