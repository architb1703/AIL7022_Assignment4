"""Reproduce post-training classifications of sparse QMIX reporting failures."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
Q1 = HERE.parents[1]
sys.path.insert(0,str(Q1))
from train_ctde import load_qmix_policy
from failure_analysis import collect_rollout, classify_failure

def main():
    torch.set_num_threads(1)
    output, totals = [], Counter()
    for seed in (0,1,2):
        name = f"qmix_sparse_seed{seed}"
        policy, _ = load_qmix_policy(Q1 / "checkpoints" / name / "final.pt")
        saved = json.loads((Q1 / "results" / name / "final_eval.json").read_text())["episodes"]
        categories, episodes = Counter(), []
        for row in saved:
            episode = collect_rollout(policy,row["eval_seed"],"sparse")
            assert episode["finish_steps"] == row["finish_steps"]
            assert episode["collisions"] == row["collisions"]
            if not episode["success"]:
                category = classify_failure(episode["states"])
                categories[category] += 1
                episodes.append({"eval_seed": row["eval_seed"], "category": category, "finish_steps": episode["finish_steps"]})
        totals.update(categories)
        finished = np.array([row["finish_steps"] for row in saved]) >= 0
        output.append({"training_seed": seed, "episodes": len(saved), "failure_categories": dict(categories), "agent_finish_rates": finished.mean(axis=0).tolist(), "classified_failures": episodes})
        print(seed,dict(categories),flush=True)
    data = {"algorithm": "qmix", "reward_mode": "sparse", "checkpoint": "final", "slip_prob": 0.1, "training_seeds": [0,1,2], "evaluation_seeds": [10000,10099], "episodes": 300, "category_priority": ["goal_blocking","deadlock","livelock","timeout"], "all_rollouts_match_saved_metrics": True, "failure_categories": dict(totals), "runs": output}
    (HERE / "qmix_sparse_failure_diagnostics.json").write_text(json.dumps(data,indent=2) + "\n")
    print("TOTAL",dict(totals))

if __name__ == "__main__":
    main()
