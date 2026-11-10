# Library Imports
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_iql import load_iql_policy
from train_iql_gru import load_iql_policy as load_gru_iql_policy
from train_ctde import load_qmix_policy
from train_ctde_non_gru import load_qmix_policy as load_non_gru_policy
from visualize import save_rollout_gif, save_static
from failure_analysis import classify_failure, collect_rollout

SEEDS = (0, 1, 2)
METRICS = ("success_rate", "makespan_success", "makespan_all", "sum_costs", "collisions", "normalized_makespan_all")
LABELS = {"iql": "IQL MLP", "iql_gru": "IQL GRU", "qmix": "QMIX GRU", "qmix_non_gru": "QMIX MLP"}

def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False, default=lambda value: value.tolist())

def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize(records, split, algorithms=("iql", "qmix")):
    rows = []
    for algorithm in algorithms:
        for reward_mode in ("sparse", "shaped"):
            group = [row for row in records if row["algorithm"] == algorithm and row["reward_mode"] == reward_mode]
            for metric in METRICS:
                values = [row[split]["summary"][metric] for row in group if row[split]["summary"][metric] is not None]
                rows.append({"algorithm": algorithm, "reward_mode": reward_mode, "metric": metric, "n_seeds_defined": len(values), "mean": float(np.mean(values)) if values else None, "std": float(np.std(values, ddof=1)) if len(values) >= 2 else None})
    return rows

def make_visualizations(best, checkpoint_dir, output_dir):
    name = best["name"]
    loader = {"iql": load_iql_policy, "iql_gru": load_gru_iql_policy, "qmix": load_qmix_policy, "qmix_non_gru": load_non_gru_policy}[best["algorithm"]]
    checkpoint = checkpoint_dir / name / "final.pt"
    policy, _ = loader(checkpoint)
    save_static(output_dir / "env_visualization.png")
    successful = [e["eval_seed"] for e in best["validation"]["episodes"] if e["success"]]
    coordination = None
    for seed in successful or [20000]:
        coordination = collect_rollout(policy,seed,best["reward_mode"])
        if coordination["success"] and coordination["coordination_events"]:
            break
    save_rollout_gif(policy,path=output_dir / "final_policy.gif",seed=coordination["seed"],slip_prob=0.1,reward_mode=best["reward_mode"])
    write_json(output_dir / "coordination_rollout.json", coordination)

    failures = []
    standard_failures = [e["eval_seed"] for e in best["final"]["episodes"] if not e["success"]]
    for seed in standard_failures:
        episode = collect_rollout(policy,seed,best["reward_mode"])
        if not episode["success"]:
            failures.append(episode)
        if len(failures) == 3:
            break
    analysis_checkpoint = checkpoint
    if len(failures) < 3:
        failures = []
        for seed in range(30000, 30200):
            episode = collect_rollout(policy,seed,best["reward_mode"],0.3)
            if not episode["success"]:
                failures.append(episode)
            if len(failures) == 3:
                break
    if len(failures) < 3:
        failures = []
        analysis_checkpoint = checkpoint_dir / name / "step_0250000.pt"
        policy, _ = loader(analysis_checkpoint)
        for seed in range(30000, 30200):
            episode = collect_rollout(policy,seed,best["reward_mode"],0.1)
            if not episode["success"]:
                failures.append(episode)
            if len(failures) == 3:
                break
    if len(failures) < 3:
        raise RuntimeError("Fewer than three failure examples found; inspect saved evaluations before extending analysis.")
    categories = ("goal_blocking", "deadlock", "livelock", "timeout")
    counts = dict.fromkeys(categories, 0)
    for i, episode in enumerate(failures, 1):
        episode["category"] = classify_failure(episode["states"])
        counts[episode["category"]] += 1
        save_rollout_gif(policy,path=output_dir / f"failure_{i}.gif",seed=episode["seed"],slip_prob=episode["slip_prob"],reward_mode=best["reward_mode"])
        write_json(output_dir / f"failure_{i}.json", episode)
    write_csv(output_dir / "failure_categories.csv", [{"category": key, "count": value, "fraction": value / len(failures)} for key, value in counts.items()])
    write_json(output_dir / "visualization_details.json", {"best_policy": name, "selection": "Final-checkpoint validation success, then makespan, sum of costs and collisions", "coordination_checkpoint": str(checkpoint), "coordination_seed": coordination["seed"], "coordination_events": coordination["coordination_events"], "failure_checkpoint": str(analysis_checkpoint), "failure_slip_prob": failures[0]["slip_prob"], "failure_seeds": [e["seed"] for e in failures], "failure_table_scope": "Fractions among the three saved failure examples; not population failure rates."})

def add_optional_results(records, results_dir, output_dir, algorithms):
    for algorithm in algorithms:
        if algorithm in ("iql", "qmix"):
            continue
        for reward_mode in ("sparse", "shaped"):
            for seed in SEEDS:
                name = f"{algorithm}_{reward_mode}_seed{seed}"
                folder = results_dir / name
                row = {"name": name, "algorithm": algorithm, "reward_mode": reward_mode, "seed": seed, "curve": read_json(folder / "learning_curve.json")}
                if [point["step"] for point in row["curve"]] != list(range(50000,1000001,50000)):
                    raise ValueError(f"Expected 20 complete evaluation checkpoints: {name}.")
                for split, seeds in (("validation",list(range(20000,20050))), ("final",list(range(10000,10100)))):
                    row[split] = read_json(folder / f"{split}_eval.json")
                    if row[split]["step"] != 1000000 or [e["eval_seed"] for e in row[split]["episodes"]] != seeds:
                        raise ValueError(f"Incomplete {split} evaluation: {name}.")
                records.append(row)
    per_seed = [{"algorithm": row["algorithm"], "reward_mode": row["reward_mode"], "seed": row["seed"], "split": split, **{key: row[split]["summary"][key] for key in METRICS}} for row in records for split in ("validation", "final")]
    write_csv(output_dir / "extended_per_seed_metrics.csv",per_seed)
    for split in ("validation", "final"):
        rows = summarize(records,split,algorithms)
        write_csv(output_dir / f"extended_{split}_metrics.csv",rows)
        lines = ["| Method | Reward | Success | Makespan (success) | Makespan (all) | Total cost | Collisions | Normalized makespan |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for algorithm in algorithms:
            for reward_mode in ("sparse", "shaped"):
                formatted = []
                for metric in METRICS:
                    value = next(row for row in rows if row["algorithm"] == algorithm and row["reward_mode"] == reward_mode and row["metric"] == metric)
                    formatted.append("undefined" if value["mean"] is None else f"{value['mean']:.3f} ± {value['std']:.3f}" if value["std"] is not None else f"{value['mean']:.3f} (one defined seed)")
                lines.append(f"| {LABELS[algorithm]} | {reward_mode} | " + " | ".join(formatted) + " |")
        (output_dir / f"extended_{split}_metrics.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(9,5))
    curve_rows = []
    for algorithm in algorithms:
        for reward_mode in ("sparse", "shaped"):
            group = [row for row in records if row["algorithm"] == algorithm and row["reward_mode"] == reward_mode]
            steps = [point["step"] for point in group[0]["curve"]]
            values = np.array([[point["success_rate"] for point in row["curve"]] for row in group])
            mean, std = values.mean(axis=0), values.std(axis=0,ddof=1)
            style = {"iql": "-", "iql_gru": ":", "qmix": "--", "qmix_non_gru": "-."}[algorithm]
            line, = ax.plot(steps,mean,label=f"{LABELS[algorithm]} {reward_mode}",linestyle=style)
            ax.fill_between(steps,np.maximum(0,mean - std),np.minimum(1,mean + std),color=line.get_color(),alpha=0.12)
            curve_rows.extend({"algorithm": algorithm, "reward_mode": reward_mode, "step": int(t), "success_mean": float(avg), "success_std": float(dev)} for t, avg, dev in zip(steps,mean,std))
    ax.set(xlabel="Joint environment steps",ylabel="Success rate",title="IQL and QMIX variants (3 training seeds)",ylim=(0,1.02))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "all_learning_curves.png",dpi=300)
    fig.savefig(output_dir / "all_learning_curves.pdf")
    plt.close(fig)
    write_csv(output_dir / "all_learning_curves.csv",curve_rows)

def main(results_dir, checkpoint_dir, output_dir, make_gifs=True, include_non_gru=False, include_iql_gru=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    records, per_seed, curve_rows = [], [], []
    fig, ax = plt.subplots(figsize=(8, 5))
    for algorithm in ("iql", "qmix"):
        for reward_mode in ("sparse", "shaped"):
            group = []
            for seed in SEEDS:
                name = f"{algorithm}_{reward_mode}_seed{seed}"
                folder = results_dir / name
                row = {"name": name, "algorithm": algorithm, "reward_mode": reward_mode, "seed": seed, "curve": read_json(folder / "learning_curve.json")}
                if [point["step"] for point in row["curve"]] != list(range(50000, 1000001, 50000)):
                    raise ValueError(f"Expected 20 complete evaluation checkpoints: {name}.")
                for split, seeds in (("validation", list(range(20000,20050))), ("final", list(range(10000,10100)))):
                    row[split] = read_json(folder / f"{split}_eval.json")
                    if row[split]["step"] != 1_000_000 or [episode["eval_seed"] for episode in row[split]["episodes"]] != seeds:
                        raise ValueError(f"Incomplete {split} evaluation: {name}.")
                    per_seed.append({"algorithm": algorithm, "reward_mode": reward_mode, "seed": seed, "split": split, **{key: row[split]["summary"][key] for key in METRICS}})
                records.append(row)
                group.append(row)
            steps = np.array([point["step"] for point in group[0]["curve"]])
            values = np.array([[point["success_rate"] for point in row["curve"]] for row in group])
            mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
            line, = ax.plot(steps,mean,label=f"{LABELS[algorithm]} {reward_mode}")
            ax.fill_between(steps,np.maximum(0,mean - std),np.minimum(1,mean + std),color=line.get_color(),alpha=0.18)
            curve_rows.extend({"algorithm": algorithm, "reward_mode": reward_mode, "step": int(step), "success_mean": float(avg), "success_std": float(dev)} for step, avg, dev in zip(steps, mean, std))
    ax.set(xlabel="Joint environment steps", ylabel="Success rate", title="IQL and QMIX: mean ± sample std over 3 training seeds", ylim=(0,1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png",dpi=300)
    fig.savefig(output_dir / "learning_curves.pdf")
    plt.close(fig)
    write_csv(output_dir / "learning_curves.csv", curve_rows)
    write_csv(output_dir / "per_seed_metrics.csv", per_seed)
    for split in ("validation", "final"):
        rows = summarize(records,split)
        write_csv(output_dir / f"{split}_metrics.csv", rows)
        lines = ["| Method | Reward | Success | Makespan (success) | Makespan (all) | Total cost | Collisions | Normalized makespan |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for algorithm in ("iql", "qmix"):
            for reward_mode in ("sparse", "shaped"):
                formatted = []
                for metric in METRICS:
                    value = next(row for row in rows if row["algorithm"] == algorithm and row["reward_mode"] == reward_mode and row["metric"] == metric)
                    formatted.append("undefined" if value["mean"] is None else f"{value['mean']:.3f} ± {value['std']:.3f}" if value["std"] is not None else f"{value['mean']:.3f} (one defined seed)")
                lines.append(f"| {LABELS[algorithm]} | {reward_mode} | " + " | ".join(formatted) + " |")
        (output_dir / f"{split}_metrics.md").write_text("\n".join(lines) + "\n")
    if include_non_gru or include_iql_gru:
        algorithms = ("iql",) + (("iql_gru",) if include_iql_gru else ()) + ("qmix",) + (("qmix_non_gru",) if include_non_gru else ())
        add_optional_results(records,results_dir,output_dir,algorithms)
    rank = lambda row: (-row["validation"]["summary"]["success_rate"], row["validation"]["summary"]["makespan_all"], row["validation"]["summary"]["sum_costs"], row["validation"]["summary"]["collisions"])
    best = min(records,key=rank)
    write_json(output_dir / "selected_policy.json", {"name": best["name"], "checkpoint": str(checkpoint_dir / best["name"] / "final.pt"), "validation": best["validation"]["summary"], "selection_uses_reporting_seeds": False})
    if make_gifs:
        make_visualizations(best,checkpoint_dir,output_dir)
    print((output_dir / "final_metrics.md").read_text())
    print("Validation-selected policy:",best["name"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate the 12 primary Q1 runs and optional comparisons; reproduce figures and GIFs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/report"))
    parser.add_argument("--no-gifs", action="store_true")
    parser.add_argument("--include-non-gru", action="store_true", help="Also include six archived or reproduced feed-forward QMIX runs.")
    parser.add_argument("--include-iql-gru", action="store_true", help="Also include six GRU IQL runs with matched QMIX settings.")
    args = parser.parse_args()
    main(args.results_dir,args.checkpoint_dir,args.output_dir,not args.no_gifs,args.include_non_gru,args.include_iql_gru)
