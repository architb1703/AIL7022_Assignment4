# Library Imports
import argparse
import csv
import json
import importlib.metadata
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import minari
import numpy as np
import torch
from torch import nn

DATASET = "mujoco/hopper/medium-v0"
OBS_DIM, ACTION_DIM = 11, 3
TRAIN_EVAL_SEEDS = tuple(range(10000, 10010))
EVAL_SEEDS = tuple(range(20000, 20020))

# Experiment hyperparameters
@dataclass(frozen=True)
class Config:
    seed: int
    results_dir: Path
    checkpoint_dir: Path
    data_dir: Path
    total_steps: int = int(1e6)
    eval_every: int = int(5e4)
    batch_size: int = 128
    hidden_dims: tuple[int, int] = (128, 128)
    gamma: float = 0.99
    expectile: float = 0.7
    temperature: float = 3.0  # Inverse temperature: exp(temperature * advantage)
    max_weight: float = 100.0
    learning_rate: float = 3e-4
    target_update_rate: float = 0.005
    resume: bool = False

def parse_config():
    parser = argparse.ArgumentParser(description="Offline Implicit Q-Learning on Minari Hopper.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=int(1e6))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_name = f"iql_seed{args.seed}"
    return Config(seed=args.seed, total_steps=args.total_steps, data_dir=args.data_dir,
                  results_dir=args.results_dir / run_name,
                  checkpoint_dir=args.checkpoint_dir / run_name, resume=args.resume)

def init_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)

def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False, default=str)

class OfflineDataset:
    def __init__(self, dataset):
        rows = {name: [] for name in ("obs", "actions", "rewards", "next_obs", "terminals")}
        episode_returns, truncations = [], 0
        for episode in dataset.iterate_episodes():
            length = len(episode.actions)
            rows["obs"].append(episode.observations[:-1])
            rows["next_obs"].append(episode.observations[1:])
            rows["actions"].append(episode.actions)
            rows["rewards"].append(episode.rewards)
            rows["terminals"].append(episode.terminations)
            episode_returns.append(float(episode.rewards.sum()))
            truncations += int(episode.truncations.sum())

        self.data = {key: torch.from_numpy(np.concatenate(value).astype(np.float32)) for key, value in rows.items()}

        self.reward_scale = 1000.0 / (max(episode_returns) - min(episode_returns))
        self.data["rewards"] *= self.reward_scale
        self.size = len(self.data["rewards"])

        self.info = {"dataset": DATASET, "transitions": self.size, "episodes": len(episode_returns),
                     "shapes": {key: list(value.shape) for key, value in self.data.items()},
                     "terminations": int(self.data["terminals"].sum()), "truncations": truncations,
                     "episode_return_min": min(episode_returns), "episode_return_max": max(episode_returns),
                     "reward_scale": self.reward_scale, "observation_normalization": False}

    def sample(self, batch_size, rng):
        idxs = torch.from_numpy(rng.integers(self.size, size=batch_size))
        return {key: value.index_select(0, idxs) for key, value in self.data.items()}

class Network(nn.Module):
    def __init__(self, inp_dim, out_dim=1, hidden_dims=(128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], out_dim)
        )

    def forward(self, x):
        return self.net(x)

class QNetwork(nn.Module):
    def __init__(self, hidden_dims=(128, 128)):
        super().__init__()
        self.q1 = Network(OBS_DIM + ACTION_DIM, hidden_dims=hidden_dims)
        self.q2 = Network(OBS_DIM + ACTION_DIM, hidden_dims=hidden_dims)

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

class Policy(nn.Module):
    def __init__(self, hidden_dims=(128, 128)):
        super().__init__()
        self.mean = Network(OBS_DIM, ACTION_DIM, hidden_dims)
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def forward(self, obs):
        return self.mean(obs).tanh(), self.log_std.clamp(-5.0, 2.0)

def expectile_loss(diff, expectile):
    weights = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weights * diff.square()).mean()

def update_iql(online_q, target_q, value, policy, optimizers, batch, config, step):
    obs, actions = batch["obs"], batch["actions"]
    with torch.no_grad():
        q1, q2 = target_q(obs, actions)
        target_q_values = torch.minimum(q1, q2)

    value_loss = expectile_loss(target_q_values - value(obs).squeeze(-1), config.expectile)
    optimizers["value"].zero_grad(set_to_none=True)
    value_loss.backward()
    optimizers["value"].step()

    with torch.no_grad():
        advantage = target_q_values - value(obs).squeeze(-1)
        weights = (config.temperature * advantage).clamp(max=math.log(config.max_weight)).exp()
        targets = batch["rewards"] + config.gamma * (1.0 - batch["terminals"]) * value(batch["next_obs"]).squeeze(-1)

    mean, log_std = policy(obs)
    log_probs = -0.5 * (((actions - mean) * (-log_std).exp()).square() + 2.0 * log_std + math.log(2.0 * math.pi)).sum(dim=-1)
    policy_loss = -(weights * log_probs).mean()
    
    optimizers["policy"].param_groups[0]["lr"] = config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / config.total_steps))
    optimizers["policy"].zero_grad(set_to_none=True)
    policy_loss.backward()
    optimizers["policy"].step()

    q1, q2 = online_q(obs, actions)
    critic_loss = ((q1 - targets).square() + (q2 - targets).square()).mean()
    optimizers["critic"].zero_grad(set_to_none=True)
    critic_loss.backward()
    optimizers["critic"].step()
    with torch.no_grad():
        for target, online in zip(target_q.parameters(), online_q.parameters()):
            target.lerp_(online, config.target_update_rate)

    return {"critic_loss": critic_loss.item(), "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(), "advantage": advantage.mean().item(),
            "awr_weight": weights.mean().item()}

@torch.no_grad()
def choose_action(policy, obs):
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mean, _ = policy(x)
    return mean[0].numpy()

def evaluate_policy(policy, env, seeds):
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        episode_return, length = 0.0, 0
        while True:
            obs, reward, terminated, truncated, _ = env.step(choose_action(policy, obs))
            episode_return += float(reward)
            length += 1
            if terminated or truncated:
                break
        episodes.append({"eval_seed": seed, "return": episode_return, "length": length})
    returns = [row["return"] for row in episodes]
    return {"mean_return": float(np.mean(returns)), "episode_std": float(np.std(returns, ddof=1)),
            "n_episodes": len(episodes)}, episodes

def save_checkpoint(path, models, optimizers, config, rng, step, data_info):
    saved_config = {key: str(setting) if isinstance(setting, Path) else setting for key, setting in asdict(config).items()}
    torch.save({"step": step, "config": saved_config, "data_info": data_info,
                "models": {key: model.state_dict() for key, model in models.items()},
                "optimizers": {key: optimizer.state_dict() for key, optimizer in optimizers.items()},
                "torch_rng": torch.get_rng_state(), "sampling_rng": rng.bit_generator.state}, path)

def load_iql_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = Policy(tuple(checkpoint["config"]["hidden_dims"]))
    policy.load_state_dict(checkpoint["models"]["policy"])
    policy.eval()
    return policy, checkpoint

def main(config):
    torch.set_num_threads(1)
    rng = init_rng(config.seed)
    os.environ["MINARI_DATASETS_PATH"] = str(config.data_dir.resolve())
    dataset = minari.load_dataset(DATASET)
    buffer = OfflineDataset(dataset)
    env = dataset.recover_environment(eval_env=True)
    online_q, value, policy = QNetwork(config.hidden_dims), Network(OBS_DIM, hidden_dims=config.hidden_dims), Policy(config.hidden_dims)
    target_q = deepcopy(online_q).requires_grad_(False)
    models = {"online_q": online_q, "target_q": target_q, "value": value, "policy": policy}
    optimizers = {key: torch.optim.Adam(model.parameters(), lr=config.learning_rate)
                  for key, model in (("critic", online_q), ("value", value), ("policy", policy))}

    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if (config.results_dir / "config.json").exists() and not config.resume:
        raise FileExistsError("Run already exists; choose fresh output directories or use --resume.")
    start_step, curves = 0, []
    if config.resume:
        saved = torch.load(config.checkpoint_dir / "latest.pt", map_location="cpu", weights_only=True)
        for key, setting in asdict(config).items():
            if key not in ("resume", "results_dir", "checkpoint_dir", "data_dir") and saved["config"][key] != setting:
                raise ValueError(f"Resume configuration changed: {key}.")
        for key, model in models.items():
            model.load_state_dict(saved["models"][key])
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][key])
        torch.set_rng_state(saved["torch_rng"])
        rng.bit_generator.state = saved["sampling_rng"]
        start_step = saved["step"]
        curves = [row for row in json.loads((config.results_dir / "learning_curve.json").read_text()) if row["step"] <= start_step]
        loss_path = config.results_dir / "losses.csv"
        with loss_path.open() as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["step"]) <= start_step]
        with loss_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "critic_loss", "value_loss", "policy_loss", "advantage", "awr_weight"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        write_json(config.results_dir / "config.json", {**asdict(config), "dataset": DATASET,
                   "learning_curve_eval_seeds": TRAIN_EVAL_SEEDS, "final_eval_seeds": EVAL_SEEDS,
                   "versions": {name: importlib.metadata.version(name) for name in
                                ("torch", "numpy", "minari", "gymnasium", "mujoco")}})
        write_json(config.results_dir / "dataset.json", buffer.info)

    print(json.dumps(buffer.info), flush=True)
    started = time.monotonic()
    try:
        with (config.results_dir / "losses.csv").open("a" if config.resume else "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step", "critic_loss", "value_loss", "policy_loss", "advantage", "awr_weight"])
            if not config.resume:
                writer.writeheader()
            for step in range(start_step + 1, config.total_steps + 1):
                losses = update_iql(online_q, target_q, value, policy, optimizers, buffer.sample(config.batch_size, rng), config, step)
                if not all(math.isfinite(loss) for loss in losses.values()):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {losses}")
                writer.writerow({"step": step, **losses})
                if step % config.eval_every == 0 or step == config.total_steps:
                    handle.flush()
                    summary, episodes = evaluate_policy(policy, env, TRAIN_EVAL_SEEDS)
                    curves.append({"step": step, **summary})
                    write_json(config.results_dir / "learning_curve.json", curves)
                    write_json(config.results_dir / f"eval_{step:07d}.json", {"step": step, "summary": summary, "episodes": episodes})
                    save_checkpoint(config.checkpoint_dir / "latest.pt", models, optimizers, config, rng, step, buffer.info)
                    elapsed = time.monotonic() - started
                    print(f"seed={config.seed} step={step} return={summary['mean_return']:.2f} steps/s={(step-start_step)/elapsed:.1f} losses={losses}", flush=True)

        save_checkpoint(config.checkpoint_dir / "final.pt", models, optimizers, config, rng, config.total_steps, buffer.info)
        summary, episodes = evaluate_policy(policy, env, EVAL_SEEDS)
        write_json(config.results_dir / "final_eval.json", {"step": config.total_steps, "summary": summary, "episodes": episodes})
        print("Final evaluation:", json.dumps(summary), flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    main(parse_config())
