#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_root="${1:-$script_dir/../experiments/q1_run_$(TZ=Asia/Kolkata date +%Y%m%d_%H%M%S)}"
if [[ "$run_root" != /* ]]; then
    run_root="$PWD/$run_root"
fi
if [[ -e "$run_root" ]]; then
    printf 'Output directory already exists; choose a fresh directory: %s\n' "$run_root" >&2
    exit 2
fi

cd "$script_dir"
mkdir -p "$(dirname "$run_root")"
mkdir "$run_root"

workers="${RL_WORKERS:-4}"
include_non_gru="${INCLUDE_NON_GRU:-0}"
include_iql_gru="${INCLUDE_IQL_GRU:-0}"
if [[ "$include_non_gru" != 0 && "$include_non_gru" != 1 ]]; then
    printf 'INCLUDE_NON_GRU must be 0 or 1.\n' >&2
    exit 2
fi
if [[ "$include_iql_gru" != 0 && "$include_iql_gru" != 1 ]]; then
    printf 'INCLUDE_IQL_GRU must be 0 or 1.\n' >&2
    exit 2
fi
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RL_WORKERS must be a positive integer.\n' >&2
    exit 2
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$run_root/cache/matplotlib" XDG_CACHE_HOME="$run_root/cache"

pids=()
active=0

cleanup() {
    for pid in $(jobs -pr); do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_batch() {
    for ((idx=0; idx<active; idx++)); do
        wait "${pids[$idx]}"
    done
    pids=()
    active=0
}

track_job() {
    pids[active]="$1"
    active=$((active + 1))
    if [[ "$active" -ge "$workers" ]]; then
        wait_batch
    fi
}

run_count=$((12 + 6*include_non_gru + 6*include_iql_gru))
printf 'Running all %s experiments, up to %s at a time. Outputs: %s\n' "$run_count" "$workers" "$run_root"
for seed in 0 1 2; do
    for reward_mode in sparse shaped; do
        iql_run="iql_${reward_mode}_seed${seed}"
        qmix_run="qmix_${reward_mode}_seed${seed}"
        mkdir -p "$run_root/results/$iql_run" "$run_root/results/$qmix_run"
        common_args=(--seed "$seed" --reward-mode "$reward_mode" --total-steps 1000000 --results-dir "$run_root/results" --checkpoint-dir "$run_root/checkpoints")

        printf 'Starting %s\n' "$iql_run"
        python -u train_iql.py "${common_args[@]}" > "$run_root/results/$iql_run/console.log" 2>&1 &
        track_job "$!"

        printf 'Starting %s\n' "$qmix_run"
        python -u train_ctde.py "${common_args[@]}" > "$run_root/results/$qmix_run/console.log" 2>&1 &
        track_job "$!"

        if [[ "$include_iql_gru" == 1 ]]; then
            gru_iql_run="iql_gru_${reward_mode}_seed${seed}"
            mkdir -p "$run_root/results/$gru_iql_run"
            printf 'Starting %s\n' "$gru_iql_run"
            python -u train_iql_gru.py "${common_args[@]}" > "$run_root/results/$gru_iql_run/console.log" 2>&1 &
            track_job "$!"
        fi

        if [[ "$include_non_gru" == 1 ]]; then
            non_gru_run="qmix_non_gru_${reward_mode}_seed${seed}"
            mkdir -p "$run_root/results/$non_gru_run"
            printf 'Starting %s\n' "$non_gru_run"
            python -u train_ctde_non_gru.py "${common_args[@]}" > "$run_root/results/$non_gru_run/console.log" 2>&1 &
            track_job "$!"
        fi
    done
done
wait_batch
printf 'All training runs finished. Results: %s/results\nCheckpoints: %s/checkpoints\n' "$run_root" "$run_root"

report_args=(--results-dir "$run_root/results" --checkpoint-dir "$run_root/checkpoints" --output-dir "$run_root/artifacts")
if [[ "$include_non_gru" == 1 ]]; then
    report_args+=(--include-non-gru)
fi
if [[ "$include_iql_gru" == 1 ]]; then
    report_args+=(--include-iql-gru)
fi
python reporting.py "${report_args[@]}"
