#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
export PYTHONPATH="$script_dir/.deps_21:$script_dir/.deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$script_dir/.cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
exec python -u run_decoupled.py "$@"
