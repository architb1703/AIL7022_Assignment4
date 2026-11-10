```bash
# Run from Q1/. Checkpoints: checkpoints/<run>/; logs: results/<run>/console.log.
python -m pip install -r requirements.txt
python tests.py

# Reproduce all 24 reported runs in a fresh output directory.
INCLUDE_NON_GRU=1 INCLUDE_IQL_GRU=1 ./run.sh ../experiments/reproduction

# Recreate report outputs from the submitted checkpoints and results.
python reporting.py --results-dir results --checkpoint-dir checkpoints --output-dir results/report --include-non-gru --include-iql-gru
cp results/report/env_visualization.png env_visualization.png
cp results/report/final_policy.gif final_policy.gif
python results/report/compile_iql_results.py --motion-diagnostics
python results/report/compile_ctde_diagnostics.py
python results/report/make_coordination_strip.py
```
