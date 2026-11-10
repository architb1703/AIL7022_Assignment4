#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
workers="${RL_WORKERS:-3}"
if [[ ! "$workers" =~ ^[1-5]$ ]]; then
    printf 'RL_WORKERS must be between 1 and 5.\n' >&2
    exit 2
fi

export PYTHONPATH="$script_dir/.deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$script_dir/.cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1

cleanup() {
    for pid in $(jobs -pr); do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

pids=()
for seed in 0 1 2 3 4; do
    run="iql_seed$seed"
    if [[ -f "results/$run/final_eval.json" ]]; then
        printf 'Already finished: %s\n' "$run"
        continue
    fi
    mkdir -p "results/$run"
    args=(--seed "$seed")
    if [[ -f "checkpoints/$run/latest.pt" ]]; then
        args+=(--resume)
    fi
    python -u train_iql_offline.py "${args[@]}" >> "results/$run/console.log" 2>&1 &
    pids+=("$!")
    if [[ "${#pids[@]}" -ge "$workers" ]]; then
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        pids=()
    fi
done
if [[ "${#pids[@]}" -gt 0 ]]; then
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
fi

python -u reporting.py > results/reporting.log 2>&1
printf 'Subtask 2.2 finished. Outputs: %s/results\n' "$script_dir"
