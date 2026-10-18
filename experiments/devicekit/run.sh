#!/usr/bin/env bash
set -u
KIT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
if [ ! -f "$KIT_ROOT/experiments/devicekit/run.py" ]; then
  KIT_ROOT="$(cd -- "$KIT_ROOT/../.." && pwd)" || exit 1
fi
cd -- "$KIT_ROOT" || exit 1
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$KIT_ROOT/.runtime/python/bin/python3" ]; then
    PYTHON="$KIT_ROOT/.runtime/python/bin/python3"
  elif [ -x "$KIT_ROOT/.venv/bin/python" ]; then
    PYTHON="$KIT_ROOT/.venv/bin/python"
  else
    PYTHON=python3
  fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  mkdir -p results/bootstrap
  printf '%s\n' '{"overall":"BLOCKED","code":"E_PYTHON","reason":"Python unavailable; no model executed","measured":false}' > results/bootstrap/report.json
  printf '%s\n' '# BLOCKED: E_PYTHON' 'Python unavailable; no model executed.' > results/bootstrap/report.md
  echo 'BLOCKED E_PYTHON: results/bootstrap/report.json'; exit 2
fi
exec "$PYTHON" -m experiments.devicekit.run "$@"
