# Library Imports
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ALGORITHMS = {"td3_bc": "TD3+BC", "iql_awr": "IQL+AWR", "cql_awr": "CQL+AWR"}
FRACTIONS = (.1, .5, 1.)

def read_json(path):
    return json.loads(path.read_text())

def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main(results_dir, checkpoint_dir, datasets):
    torch.set_num_threads(1)
    rows, matrices, curves = [], {}, {}
    for dataset in datasets:
        for algorithm, label in ALGORITHMS.items():
            matrix = np.empty((3, 3))
            for i, c in enumerate(FRACTIONS):
                name = f"{dataset}_{algorithm}_c{int(100*c)}_seed0"
                family = results_dir / name
                config, complete = read_json(family/"config.json"), read_json(family/"complete.json")
                if config["critic_steps"] != 50000 or config["total_steps"] != 1000000 or not complete["critics_frozen"]:
                    raise ValueError(f"Incorrect critic budget: {name}.")
                for j, p in enumerate(FRACTIONS):
                    run = f"p{int(100*p)}"
                    final = read_json(family/run/"final_eval.json")
                    curve = read_json(family/run/"learning_curve.json")
                    checkpoint = torch.load(checkpoint_dir/name/run/"final.pt", map_location="cpu", weights_only=True)
                    if checkpoint["step"] != 950000 or final["step"] != 1000000 or len(final["episodes"]) != 20:
                        raise ValueError(f"Incomplete final policy: {name}/{run}.")
                    if [x["step"] for x in curve] != list(range(100000, 1000001, 100000)) or any(x["n_episodes"] != 5 for x in curve):
                        raise ValueError(f"Incomplete learning curve: {name}/{run}.")
                    values = np.array([x["return"] for x in final["episodes"]])
                    if not np.isfinite(values).all() or not np.isclose(values.mean(), final["summary"]["mean_return"]):
                        raise ValueError("Invalid final return summary.")
                    matrix[i, j] = values.mean()
                    curves[dataset, algorithm, i, j] = curve
                    rows.append({"dataset": dataset, "algorithm": label, "critic_fraction": c, "policy_fraction": p,
                                 "seed": 0, "mean_return": float(values.mean()), "episode_std": float(values.std(ddof=1)), "n_episodes": 20})
            matrices[dataset, algorithm] = matrix
    write_csv(results_dir/"final_returns.csv", rows)
    gains = [{"dataset": dataset, "algorithm": ALGORITHMS[algorithm],
              "critic_gain": float((matrix[-1,:]-matrix[0,:]).mean()),
              "policy_gain": float((matrix[:,-1]-matrix[:,0]).mean())}
             for (dataset, algorithm), matrix in matrices.items()]
    write_csv(results_dir/"data_gains.csv", gains)
    (results_dir/"summary.json").write_text(json.dumps({"training_seed": 0, "critic_steps": 50000, "policy_steps": 950000,
        "periodic_episodes": 5, "final_episodes": 20, "evaluation_interval": 100000, "returns": rows, "gains": gains}, indent=2))

    fig, axes = plt.subplots(len(datasets), 3, figsize=(10, 3.4*len(datasets)), squeeze=False, layout="constrained")
    for i, dataset in enumerate(datasets):
        maximum = max(1, max(matrices[dataset,a].max() for a in ALGORITHMS))
        for j, (algorithm, label) in enumerate(ALGORITHMS.items()):
            ax, matrix = axes[i,j], matrices[dataset,algorithm]
            plot = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=maximum)
            ax.set_xticks(range(3), ("10%", "50%", "100%"))
            ax.set_yticks(range(3), ("10%", "50%", "100%"))
            ax.set(xlabel="Policy data", ylabel="Critic data", title=f"{dataset.capitalize()}: {label} | seed 0")
            ax.tick_params(labelsize=12)
            ax.title.set_fontsize(12)
            ax.xaxis.label.set_size(12)
            ax.yaxis.label.set_size(12)
            for c in range(3):
                for p in range(3):
                    ax.text(p,c,f"{matrix[c,p]:.1f}",ha="center",va="center",fontsize=12,color="white" if matrix[c,p]/maximum<.5 else "black")
        bar = fig.colorbar(plot, ax=list(axes[i,:]), label="Mean raw return (20 episodes)", shrink=.75)
        bar.ax.tick_params(labelsize=11)
        bar.ax.yaxis.label.set_size(12)
    fig.suptitle("Decoupled data scaling: final policies")
    fig.savefig(results_dir/"data_scaling_matrices.png", dpi=250)
    plt.close(fig)
    for dataset in datasets:
        for algorithm, label in ALGORITHMS.items():
            fig, axes = plt.subplots(3,3,figsize=(8.4,6.4),sharex=True,sharey=True)
            for c in range(3):
                for p in range(3):
                    curve, ax = curves[dataset,algorithm,c,p], axes[c,p]
                    line, = ax.plot([x["step"] for x in curve],[x["mean_return"] for x in curve])
                    ax.set_title(f"Critic {int(100*FRACTIONS[c])}%, policy {int(100*FRACTIONS[p])}%",fontsize=9)
                    ax.tick_params(labelsize=8)
                    ax.grid(alpha=.25)
            fig.suptitle(f"{dataset.capitalize()}: {label} | training seed 0", y=.995)
            fig.legend([line],["Mean over 5 deterministic evaluation episodes"],loc="upper center",bbox_to_anchor=(.5,.965),fontsize=9)
            fig.supxlabel("Total gradient iterations (critic + policy)",fontsize=10)
            fig.supylabel("Average episodic return (raw)",fontsize=10)
            fig.tight_layout(rect=(.025,.035,1,.91))
            fig.savefig(results_dir/f"{dataset}_{algorithm}_learning_curves.png",dpi=250)
            plt.close(fig)
    print(f"Reported {len(rows)} configurations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate decoupled data-scaling matrices and learning curves.")
    parser.add_argument("--results-dir",type=Path,default=Path("results/decoupled"))
    parser.add_argument("--checkpoint-dir",type=Path,default=Path("checkpoints/decoupled"))
    parser.add_argument("--datasets",nargs="+",choices=("hopper","kitchen"),default=("hopper","kitchen"))
    args=parser.parse_args()
    main(args.results_dir,args.checkpoint_dir,args.datasets)
