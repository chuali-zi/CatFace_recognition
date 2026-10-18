import ast
import json
import zipfile
from pathlib import Path

from experiments.board.run import identity_metrics, telemetry_summary
from experiments.devicekit import suite


def test_replay_export_detects_thread_drift_without_exporting_paths(tmp_path):
    for thread, score, status in [(1, .9, 'known'), (2, .8, 'unknown')]:
        folder = tmp_path / f'small160-t{thread}'
        folder.mkdir()
        (folder / 'summary.json').write_text(json.dumps({'model': 'small160', 'threads': thread}))
        rows = [dict(sample_id='private-name.jpg', path='/private/cat.jpg', true_cat_id='owner-cat',
            cat_id='owner-cat' if status == 'known' else None, score=score, role='known',
            status=status, repeat=repeat, total_ms=20) for repeat in range(2)]
        (folder / 'measurements.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    summaries, observations, consistency = suite.collect_replay(tmp_path)
    assert len(summaries) == 2 and len(observations) == 4
    assert consistency[0]['decision_changes'] == 2
    assert abs(consistency[0]['max_score_delta'] - .1) < 1e-10
    assert {r['sample_id'] for r in observations} == {'sample-001'}
    assert 'private' not in json.dumps(observations) and 'owner-cat' not in json.dumps(observations)


def test_missing_sensors_are_absent_and_query_failures_remain_in_denominator():
    assert telemetry_summary([]) == {'temperature_c': {}, 'cpu_frequency_khz': {}, 'rss_mib': None}
    rows = [dict(role='known', status='error', true_cat_id='a'),
            dict(role='known', status='known', true_cat_id='a', cat_id='a'),
            dict(role='unknown', status='no_cat', true_cat_id='b')]
    metrics = identity_metrics(rows)
    assert metrics['known_queries'] == 2 and metrics['dir'] == .5
    assert metrics['unknown_status_counts'] == {'no_cat': 1}


def test_suite_integrity_failure_still_returns_one_safe_archive(tmp_path, monkeypatch):
    root = tmp_path / 'kit'
    root.mkdir()
    monkeypatch.setattr(suite, 'ROOT', root)
    out = root / 'results' / 'failed'
    assert suite.main(['--out', str(out)]) == 1
    report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
    assert report['overall'] == 'FAIL'
    assert (out / 'private/failure.log').exists()
    with zipfile.ZipFile(out.with_suffix('.zip')) as archive:
        assert archive.testzip() is None
        assert not any('private' in n for n in archive.namelist())
        assert {'report.json', 'report.md', 'summary.json', 'summary.csv', 'observations.jsonl'} <= set(archive.namelist())


def test_suite_can_start_with_board_system_python38():
    ast.parse(Path('experiments/devicekit/suite.py').read_text(encoding='utf-8'), feature_version=(3, 8))
