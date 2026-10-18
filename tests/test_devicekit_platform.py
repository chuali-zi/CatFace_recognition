"""Use the supplied board evidence, not host architecture, as the regression."""
import ast
import json
from pathlib import Path

from experiments.devicekit.platform_check import assess, inventory, error_summary


def board(python='3.8.10'):
    return dict(system='Linux',architecture='aarch64',implementation='CPython',
                python=python,libc=['glibc','2.31'])


def test_reported_board_requires_only_python_platform_change():
    result=assess(board())
    assert not result['compatible']
    assert [f['code'] for f in result['findings']]==['E_PYTHON_VERSION']
    assert 'bootstrap' in result['findings'][0]['remediation']
    assert assess(board('3.11.17'))['compatible']


def test_no_silent_python_or_libc_support_expansion():
    assert not assess(board('3.13.1'))['compatible']
    env=board('3.11.17'); env['libc']=['glibc','2.27']
    assert assess(env)['findings'][0]['code']=='E_LIBC'


def test_versions_are_inventory_not_import_proof():
    rows=inventory({'numpy':'1.24.4','onnxruntime':None},
                   {'numpy':'1.26.4','onnxruntime':'1.20.1'})
    assert [r['state'] for r in rows]==['PIN_MISMATCH','MISSING']
    assert not any(r['import_tested'] for r in rows)


def test_diagnostic_and_bootstrap_syntax_remains_python38_readable():
    for name in ('run','resources','platform_check','build','bootstrap'):
        ast.parse(Path(f'experiments/devicekit/{name}.py').read_text(encoding='utf-8'),feature_version=(3,8))


def test_error_classification_does_not_export_private_log():
    raw="File /private/cat.jpg, secret=do-not-export\nModuleNotFoundError: No module named 'onnxruntime'\nGLIBCXX_3.4 missing"
    result=error_summary(raw)
    assert result['missing_module']=='onnxruntime'
    assert result['error_type']=='ModuleNotFoundError'
    assert result['loader_signals']==['libstdcxx_version']
    assert 'secret' not in json.dumps(result) and '/private' not in json.dumps(result)
