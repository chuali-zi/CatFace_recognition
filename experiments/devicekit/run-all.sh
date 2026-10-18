#!/usr/bin/env bash
set -u
KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
if [ ! -f "$KIT_ROOT/experiments/devicekit/suite.py" ]; then
  KIT_ROOT="$(cd -- "$KIT_ROOT/../.." && pwd)" || exit 1
fi
cd -- "$KIT_ROOT" || exit 1
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -z "${PYTHON:-}" ]; then
  if [ -x .runtime/python/bin/python3 ]; then
    PYTHON="$KIT_ROOT/.runtime/python/bin/python3"
  elif [ -x .venv/bin/python ]; then
    PYTHON="$KIT_ROOT/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi
exec "$PYTHON" -m experiments.devicekit.suite "$@"
