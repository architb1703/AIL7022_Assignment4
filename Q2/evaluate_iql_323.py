# Library Imports
import argparse
import json
import platform
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(checkpoint_dir, results_dir):
    if mujoco.mj_versionString() != "3.2.3":
        raise RuntimeError("This validation requires MuJoCo 3.2.3 and Python 3.11.")
    torch.set_num_threads(1)
    versions = {"mujoco": mujoco.mj_versionString(), "gymnasium": gym.__version__,
                "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()}
    env = gym.make("Hopper-v5")
    rows = []
    try:
        for seed in range(5):
            saved = torch.load(checkpoint_dir/f"iql_seed{seed}"/"final.pt", map_location="cpu", weights_only=True)
            if saved["step"] != 1000000:
                raise ValueError("Use the final million-step checkpoint.")
            state = saved["models"]["policy"]
            episodes = []
            for eval_seed in range(20000, 20020):
                obs, _ = env.reset(seed=eval_seed)
                total, length = 0.0, 0
                while True:
                    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    for layer in (0, 2):
                        x = F.relu(F.linear(x, state[f"mean.net.{layer}.weight"], state[f"mean.net.{layer}.bias"]))
                    action = F.linear(x, state["mean.net.4.weight"], state["mean.net.4.bias"]).tanh()[0].numpy()
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    length += 1
                    if terminated or truncated:
                        break
                episodes.append({"eval_seed": eval_seed, "return": total, "length": length})
            values = np.array([episode["return"] for episode in episodes])
            summary = {"mean_return": float(values.mean()), "episode_std": float(values.std(ddof=1)), "n_episodes": 20}
            directory = results_dir/f"iql_seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory/"final_eval.json").write_text(json.dumps({"step": 1000000, "summary": summary,
                "episodes": episodes, "evaluation_versions": versions}, indent=2, allow_nan=False))
            rows.append({"seed": seed, **summary})
            print(f"Seed {seed}: {summary['mean_return']:.2f}", flush=True)
    finally:
        env.close()
    values = np.array([row["mean_return"] for row in rows])
    summary = {"n_seeds": 5, "mean_return": float(values.mean()), "seed_std": float(values.std(ddof=1)),
               "evaluation_mujoco": "3.2.3", "per_seed": rows}
    (results_dir/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Across seeds: {summary['mean_return']:.2f} +/- {summary['seed_std']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reevaluate the five final IQL policies on MuJoCo 3.2.3.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/validation_323"))
    args = parser.parse_args()
    evaluate(args.checkpoint_dir, args.results_dir)
