"""Reproduce the IQL-only figure, tables and optional rollout diagnostics for 1.3(b)."""
import argparse
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
METRICS = ("success_rate", "makespan_success", "makespan_all", "sum_costs", "collisions", "normalized_makespan_all")

def main(motion=False):
    summaries, curves, per_seed = {}, {}, []
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for mode in ("sparse", "shaped"):
        finals, history = [], []
        for seed in (0, 1, 2):
            folder = RESULTS / f"iql_{mode}_seed{seed}"
            data = json.loads((folder / "final_eval.json").read_text())
            curve = json.loads((folder / "learning_curve.json").read_text())
            assert data["step"] == 1000000
            assert [e["eval_seed"] for e in data["episodes"]] == list(range(10000,10100))
            assert [point["step"] for point in curve] == list(range(50000,1000001,50000))
            finals.append(data["summary"])
            history.append(curve)
            per_seed.append({"reward_mode": mode, "seed": seed, **{key: data["summary"][key] for key in METRICS}})
        summaries[mode] = {key: {"mean": float(np.mean([row[key] for row in finals])), "std": float(np.std([row[key] for row in finals],ddof=1))} for key in METRICS}
        values = np.array([[row["success_rate"] for row in curve] for curve in history])
        mean, std = values.mean(axis=0), values.std(axis=0,ddof=1)
        steps = np.array([row["step"] for row in history[0]])
        line, = ax.plot(steps,mean,label=f"IQL {mode}",linewidth=2)
        ax.fill_between(steps,np.maximum(0,mean - std),np.minimum(1,mean + std),color=line.get_color(),alpha=0.2)
        curves[mode] = [{"step": int(t), "mean": float(avg), "std": float(dev)} for t, avg, dev in zip(steps, mean, std)]
    ax.set(xlabel="Joint environment steps",ylabel="Success rate",title="IQL: sparse and shaped rewards (3 training seeds)",ylim=(0,1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "iql_learning_curves.png",dpi=300)
    fig.savefig(HERE / "iql_learning_curves.pdf")
    plt.close(fig)
    (HERE / "iql_results.json").write_text(json.dumps({"final_metrics": summaries, "learning_curves": curves, "standard_deviation": "sample standard deviation across 3 training-seed means"},indent=2) + "\n")
    with (HERE / "iql_per_seed_metrics.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    if motion:
        sys.path.insert(0,str(HERE.parents[1]))
        import torch
        from train_iql import load_iql_policy
        from failure_analysis import collect_rollout
        torch.set_num_threads(1)
        diagnostics = {}
        for mode in ("sparse", "shaped"):
            name = f"iql_{mode}_seed2"
            policy, _ = load_iql_policy(HERE.parents[1] / "checkpoints" / name / "final.pt")
            expected = json.loads((RESULTS / name / "final_eval.json").read_text())["episodes"]
            statistics = []
            for row in expected:
                episode = collect_rollout(policy,row["eval_seed"],mode)
                assert episode["finish_steps"] == row["finish_steps"]
                waits, moves, unexecuted = 0, 0, 0
                for t, actions in enumerate(episode["actions"]):
                    old, new = episode["states"][t:t + 2]
                    if old["finished"][4]:
                        continue
                    moved = not np.array_equal(old["positions"][4],new["positions"][4])
                    waits += int(actions[4] == 0)
                    moves += int(moved)
                    unexecuted += int(actions[4] != 0 and not moved)
                statistics.append([waits,moves,unexecuted,episode["finish_steps"][4]])
            averages = np.mean(statistics,axis=0)
            diagnostics[mode] = {"training_seed": 2, "agent": 4, "episodes": 100, **{key: float(value) for key, value in zip(("mean_chosen_stay_steps","mean_executed_moves","mean_unexecuted_move_steps","mean_finish_step"),averages)}, "all_finish_steps_match_saved_reporting": True}
        (HERE / "iql_motion_diagnostics.json").write_text(json.dumps(diagnostics,indent=2) + "\n")
    print(json.dumps(summaries,indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-diagnostics",action="store_true")
    args = parser.parse_args()
    main(args.motion_diagnostics)
