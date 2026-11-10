```bash
# Run from Q2.
python3.13 -m venv .venv_21
.venv_21/bin/python -m pip install -r requirements_2_1.txt
python3.13 -m venv .venv_22
.venv_22/bin/python -m pip install -r requirements.txt

export MINARI_DATASETS_PATH="$PWD/data"
.venv_21/bin/python -c "import minari; minari.download_dataset('D4RL/kitchen/complete-v2'); minari.download_dataset('mujoco/hopper/medium-v0')"

# Subtask 2.1; completed runs are skipped.
PATH="$PWD/.venv_21/bin:$PATH" RL_WORKERS=4 bash run_decoupled.sh

# Subtask 2.2; completed runs are skipped.
PATH="$PWD/.venv_22/bin:$PATH" RL_WORKERS=3 bash run.sh

# Reported MuJoCo 3.2.3 final-policy validation (requires the separate checkpoints).
python3.11 -m venv .venv_validation
.venv_validation/bin/python -m pip install -r requirements_validation.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv_validation/bin/python evaluate_iql_323.py
```
