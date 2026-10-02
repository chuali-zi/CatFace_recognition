#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10); import numpy, PIL, onnxruntime, psutil' >/dev/null 2>&1; then
    if [[ ! -d wheels ]]; then
        echo 'Missing runtime dependencies and wheels/. Prepare matching offline wheels first (see README.md).' >&2
        exit 1
    fi
    if [[ ! -x .venv/bin/python ]]; then
        "$PYTHON" -m venv .venv || { echo 'Python >=3.10 with venv/pip is required; install it before going offline.' >&2; exit 1; }
    fi
    PYTHON="$PWD/.venv/bin/python"
    "$PYTHON" -m pip install --no-index --only-binary=:all: --find-links="$PWD/wheels" -r requirements.txt || {
        echo 'Offline wheel installation failed. Check CPU architecture, Python minor version and glibc against bundle.json.' >&2
        exit 1
    }
fi
exec "$PYTHON" -m experiments.board.run --bundle "$PWD" "$@"
