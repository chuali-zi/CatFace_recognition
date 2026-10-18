"""Stdlib-only diagnostics, including on the board's original Python 3.8."""
import re


def assess(environment):
    findings = []

    def require(ok, component, actual, required, code, remedy):
        if not ok:
            findings.append(dict(component=component, actual=actual, required=required,
                                 code=code, remediation=remedy))

    require(environment.get('system') == 'Linux', 'system', environment.get('system'),
            'Linux', 'E_OS', 'Use the Linux device; Windows checks do not prove board compatibility.')
    require(environment.get('architecture') in ('aarch64', 'arm64', 'x86_64', 'AMD64'),
            'architecture', environment.get('architecture'), '64-bit ARM64 or x86_64',
            'E_ARCH', 'Use a matching 64-bit OS/runtime; do not install wheels for another CPU.')
    require(environment.get('implementation') == 'CPython', 'implementation',
            environment.get('implementation'), 'CPython', 'E_IMPLEMENTATION', 'Use CPython 3.11.')
    version = tuple(int(x) for x in re.findall(r'\d+', environment.get('python', ''))[:2])
    require((3, 10) <= version <= (3, 12), 'python', environment.get('python'),
            '3.10–3.12 (recommended 3.11)', 'E_PYTHON_VERSION',
            'On the reported Ubuntu20.04 ARM64 device run: python3 -m experiments.devicekit.bootstrap; '
            'then bash run.sh --mode preflight. This installs only package-local .runtime, not system Python.')
    libc = environment.get('libc', ['', ''])
    libc_version = tuple(int(x) for x in re.findall(r'\d+', libc[1])[:2]) if len(libc) > 1 else ()
    require(len(libc) == 2 and libc[0] == 'glibc' and libc_version >= (2, 28),
            'libc', libc, 'glibc >= 2.28 for the included Pillow/ORT wheels', 'E_LIBC',
            'Inspect ldd --version; do not replace system libc. A different OS needs a separately tested runtime.')
    return {'compatible': not findings, 'findings': findings,
            'meaning': 'Candidate runtime requirements only; imports, actual CPU model load and board inference remain separate checks.'}


def inventory(versions, pins):
    return [{'package': name, 'installed': versions.get(name), 'required': pin,
             'state': 'MISSING' if versions.get(name) is None else
                      'PIN_MATCH' if versions[name] == pin else 'PIN_MISMATCH',
             'import_tested': False} for name, pin in pins.items()]


def error_summary(text):
    """Export fixed classifications, never raw paths/tracebacks or log contents."""
    result={}
    missing=re.search(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'",text)
    if missing: result['missing_module']=missing.group(1)
    types=re.findall(r'^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception)):',text,re.MULTILINE)
    if types: result['error_type']=types[-1]
    markers={'GLIBC_':'glibc_version','GLIBCXX_':'libstdcxx_version','CXXABI_':'cxx_abi',
             'cannot open shared object':'missing_shared_library','undefined symbol':'undefined_symbol',
             'Illegal instruction':'cpu_instruction'}
    result['loader_signals']=[value for token,value in markers.items() if token in text]
    result['raw_log_exported']=False
    return result
