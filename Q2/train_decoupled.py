# Library Imports
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import gymnasium_robotics
import minari
import numpy as np
import torch
from torch import nn

DATASETS = {"hopper": "mujoco/hopper/medium-v0", "kitchen": "D4RL/kitchen/complete-v2"}
FRACTIONS = (0.1, 0.5, 1.0)
CURVE_SEEDS = tuple(range(10000, 10005))
FINAL_SEEDS = tuple(range(20000, 20020))
CRITIC_FIELDS = ("step", "critic_loss", "td_loss", "value_loss", "auxiliary_loss", "conservative_loss", "bc_loss", "entropy")
POLICY_FIELDS = ("step", "policy_loss", "bc_loss")

# Experiment hyperparameters
@dataclass(frozen=True)
class Config:
    algorithm: str
    dataset: str
    critic_fraction: float
    seed: int = 0
    data_dir: Path = Path("data")
    results_dir: Path = Path("results/decoupled")
    checkpoint_dir: Path = Path("checkpoints/decoupled")
    policy_fractions: tuple = FRACTIONS
    total_steps: int = int(1e6)
    critic_steps: int = int(5e4)
    eval_every: int = int(1e5)
    batch_size: int = 128
    hidden_dims: tuple = (128, 128)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    target_update_rate: float = 0.005
    expectile: float = 0.7
    temperature: float = 3.0
    max_weight: float = 100.0
    bc_alpha: float = 2.5
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    conservative_weight: float = 5.0
    n_action_samples: int = 10
    entropy_temperature: float = 0.2
    soft_q_backup: bool = False
    profile: str = "baseline"
    normalize_observations: bool = False
    actor_dropout: float = 0.0

def parse_config():
    parser = argparse.ArgumentParser(description="Decoupled TD3+BC, IQL+AWR and CQL+AWR data scaling.")
    parser.add_argument("--algorithm", choices=("td3_bc", "iql_awr", "cql_awr"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--critic-fraction", type=float, choices=FRACTIONS, required=True)
    parser.add_argument("--policy-fractions", type=float, nargs="+", choices=FRACTIONS, default=FRACTIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=("baseline", "kitchen_v2"), default="baseline")
    parser.add_argument("--critic-steps", type=int, default=int(5e4))
    parser.add_argument("--total-steps", type=int, default=int(1e6))
    parser.add_argument("--eval-every", type=int, default=int(1e5))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/decoupled"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/decoupled"))
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or not 0 < args.critic_steps < args.total_steps or args.eval_every <= 0:
        parser.error("Use a valid seed and 0 < critic steps < total steps, with positive evaluation interval.")
    args.policy_fractions = tuple(args.policy_fractions)
    return Config(**vars(args))

def apply_profile(config):
    if config.profile == "kitchen_v2":
        return replace(config, normalize_observations=True, temperature=0.5,
                       actor_dropout=0.0 if config.algorithm == "td3_bc" else 0.1)
    return config

def init_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=str))
    temporary.replace(path)

def settings(config):
    return json.loads(json.dumps(asdict(config), default=str))

def check_config(saved, config):
    for key, value in settings(config).items():
        if key not in ("data_dir", "results_dir", "checkpoint_dir", "policy_fractions") and saved.get(key, getattr(Config, key, None)) != value:
            raise ValueError(f"Existing run has a different {key}.")

def encode_obs(obs):
    return np.asarray(obs["observation"] if isinstance(obs, dict) else obs, dtype=np.float32)

# Explicit transitions preserve next observations and time-limit bootstrapping.
class OfflineDataset:
    def __init__(self, dataset):
        rows = {key: [] for key in ("obs", "actions", "rewards", "next_obs", "terminals")}
        truncations = 0
        for episode in dataset.iterate_episodes():
            obs, actions = encode_obs(episode.observations), episode.actions
            length = len(actions)
            for key, value in zip(rows, (obs[:-1], actions, episode.rewards, obs[1:], episode.terminations)):
                rows[key].append(value)
            truncations += int(episode.truncations.sum())
        self.data = {key: torch.from_numpy(np.concatenate(value).astype(np.float32)) for key, value in rows.items()}
        self.size = len(self.data["rewards"])
        self.obs_dim, self.action_dim = self.data["obs"].shape[1], self.data["actions"].shape[1]

        self.info = {"dataset": dataset.spec.dataset_id, "transitions": self.size, "episodes": dataset.total_episodes,
                     "shapes": {key: list(value.shape) for key, value in self.data.items()},
                     "terminations": int(self.data["terminals"].sum()), "truncations": truncations,
                     "observation_encoding": "observation field" if self.obs_dim == 59 else "vector",
                     "observation_normalization": False, "reward_scale": 1.0}

    def sample(self, indices, batch_size, rng):
        idxs = torch.from_numpy(indices[rng.integers(len(indices), size=batch_size)])
        return {key: value.index_select(0, idxs) for key, value in self.data.items()}

    def observation_stats(self, indices):
        obs = self.data["obs"].index_select(0, torch.from_numpy(indices))
        return obs.mean(0), obs.std(0, unbiased=False) + 1e-3

def subset_permutations(config, size):
    directory = config.results_dir / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{config.dataset}_seed{config.seed}.npz"
    if not path.exists():
        seeds = np.random.SeedSequence(config.seed).spawn(2)
        critic, policy = [np.random.default_rng(seed).permutation(size) for seed in seeds]
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.npz")
        np.savez(temporary, critic=critic, policy=policy)
        temporary.replace(path)
    with np.load(path) as saved:
        permutations = {key: saved[key] for key in ("critic", "policy")}

    return permutations, path

class ObservationNormalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, obs):
        return (obs - self.mean) / self.std

class Network(nn.Module):
    def __init__(self, inp_dim, out_dim=1, hidden_dims=(128, 128), normalization=None, dropout=0.0):
        super().__init__()
        self.normalizer = nn.Identity() if normalization is None else ObservationNormalizer(*normalization)
        layers = [nn.Linear(inp_dim, hidden_dims[0]), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.normalizer(x))

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(128, 128), normalization=None):
        super().__init__()
        self.q1 = Network(obs_dim + action_dim, hidden_dims=hidden_dims)
        self.q2 = Network(obs_dim + action_dim, hidden_dims=hidden_dims)
        self.normalizer = nn.Identity() if normalization is None else ObservationNormalizer(*normalization)

    def forward(self, obs, actions):
        x = torch.cat([self.normalizer(obs), actions], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def first(self, obs, actions):
        return self.q1(torch.cat([self.normalizer(obs), actions], dim=-1)).squeeze(-1)

class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(128, 128), normalization=None, dropout=0.0):
        super().__init__()
        self.mean = Network(obs_dim, action_dim, hidden_dims, normalization, dropout)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        return self.mean(obs).tanh(), self.log_std.clamp(-5.0, 2.0)

def gaussian_log_prob(policy, obs, actions):
    mean, log_std = policy(obs)
    return -0.5 * (((actions - mean) * (-log_std).exp()).square() + 2 * log_std + math.log(2 * math.pi)).sum(-1)

def sample_squashed(policy, obs):
    mean = policy.mean(obs)
    log_std = policy.log_std.clamp(-5.0, 2.0)
    noise = torch.randn_like(mean)
    raw = mean + log_std.exp() * noise
    log_prob = (-0.5 * noise.square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    correction = (2 * (math.log(2) - raw - nn.functional.softplus(-2 * raw))).sum(-1)
    return raw.tanh(), log_prob - correction

def expectile_loss(diff, expectile):
    return (torch.where(diff > 0, expectile, 1 - expectile) * diff.square()).mean()

@torch.no_grad()
def polyak(target, online, rate):
    for a, b in zip(target.parameters(), online.parameters()):
        a.lerp_(b, rate)

def td3_policy_loss(q, policy, obs, actions, alpha):
    proposed = policy(obs)[0]
    q_values = q.first(obs, proposed)
    scale = alpha / q_values.abs().mean().detach().clamp_min(1e-6)
    bc_loss = (proposed - actions).square().mean()
    return -scale * q_values.mean() + bc_loss, bc_loss

def conservative_loss(q, policy, batch, data_q, n_samples):
    obs = batch["obs"].repeat_interleave(n_samples, dim=0)
    next_obs = batch["next_obs"].repeat_interleave(n_samples, dim=0)
    size, action_dim = len(batch["obs"]), batch["actions"].shape[1]
    with torch.no_grad():
        random_actions = torch.rand(len(obs), action_dim) * 2 - 1
        current_actions, current_logp = sample_squashed(policy, obs)
        next_actions, next_logp = sample_squashed(policy, next_obs)
        logp = torch.cat([torch.full_like(current_logp, -action_dim * math.log(2)), current_logp, next_logp])
    proposed = torch.cat([random_actions, current_actions, next_actions])
    estimates = q(torch.cat([obs, obs, obs]), proposed)
    losses = []
    for estimate, observed in zip(estimates, data_q):
        values = (estimate - logp).reshape(3, size, n_samples).permute(1, 0, 2).reshape(size, 3 * n_samples)
        losses.append((values.logsumexp(-1) - observed).mean())
    return sum(losses)

def update_critic(models, optimizers, batch, config, step):
    q, target, value, auxiliary = [models[key] for key in ("q", "target_q", "value", "auxiliary")]
    obs, actions = batch["obs"], batch["actions"]
    value_loss = auxiliary_loss = penalty = bc_loss = entropy = None
    if config.algorithm != "td3_bc":
        with torch.no_grad():
            target_values = torch.minimum(*target(obs, actions))
        value_loss = expectile_loss(target_values - value(obs).squeeze(-1), config.expectile)
        optimizers["value"].zero_grad(set_to_none=True)
        value_loss.backward()
        optimizers["value"].step()

    with torch.no_grad():
        if config.algorithm == "iql_awr":
            next_value = value(batch["next_obs"]).squeeze(-1)
        else:
            next_actions = (models["target_policy"] if config.algorithm == "td3_bc" else auxiliary)(batch["next_obs"])[0]
            if config.algorithm == "td3_bc":
                noise = (torch.randn_like(next_actions) * config.policy_noise).clamp(-config.noise_clip, config.noise_clip)
                next_actions = (next_actions + noise).clamp(-1, 1)
            next_value = torch.minimum(*target(batch["next_obs"], next_actions))
        targets = batch["rewards"] + config.gamma * (1 - batch["terminals"]) * next_value

    data_q = q(obs, actions)
    td_loss = sum((estimate - targets).square().mean() for estimate in data_q)
    loss = td_loss
    if config.algorithm == "cql_awr":
        penalty = conservative_loss(q, auxiliary, batch, data_q, config.n_action_samples)
        loss = loss + config.conservative_weight * penalty
    optimizers["q"].zero_grad(set_to_none=True)
    loss.backward()
    optimizers["q"].step()

    if config.algorithm == "cql_awr" or (config.algorithm == "td3_bc" and step % config.policy_delay == 0):
        q.requires_grad_(False)
        if config.algorithm == "td3_bc":
            auxiliary_loss, bc_loss = td3_policy_loss(q, auxiliary, obs, actions, config.bc_alpha)
        else:
            proposed, log_prob = sample_squashed(auxiliary, obs)
            auxiliary_loss = (config.entropy_temperature * log_prob - torch.minimum(*q(obs, proposed))).mean()
            entropy = -log_prob.mean()
        optimizers["auxiliary"].zero_grad(set_to_none=True)
        auxiliary_loss.backward()
        optimizers["auxiliary"].step()
        q.requires_grad_(True)
    if config.algorithm != "td3_bc" or step % config.policy_delay == 0:
        polyak(target, q, config.target_update_rate)
        if config.algorithm == "td3_bc":
            polyak(models["target_policy"], auxiliary, config.target_update_rate)
    losses = dict(zip(CRITIC_FIELDS[1:], (loss, td_loss, value_loss, auxiliary_loss, penalty, bc_loss, entropy)))
    return {key: item.item() if item is not None else None for key, item in losses.items()}

def model_digest(models):
    digest = hashlib.sha256()
    for name, model in models.items():
        for key, tensor in model.state_dict().items():
            digest.update((name + key).encode())
            digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()

def trim_losses(path, step):
    if not path.exists():
        return
    temporary = path.with_suffix(".tmp")
    with path.open() as source, temporary.open("w") as destination:
        destination.write(next(source))
        for row in source:
            if int(row.split(",", 1)[0]) > step:
                break
            destination.write(row)
    temporary.replace(path)

def save_state(path, models, optimizers, config, rng, step, **extra):
    temporary = path.with_suffix(".tmp")
    torch.save({"step": step, "config": settings(config),
                "models": {key: model.state_dict() for key, model in models.items()},
                "optimizers": {key: optimizer.state_dict() for key, optimizer in optimizers.items()},
                "torch_rng": torch.get_rng_state(), "sampling_rng": rng.bit_generator.state, **extra}, temporary)
    temporary.replace(path)

def restore(path, models, optimizers, config, rng):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    check_config(saved["config"], config)
    for key, model in models.items():
        model.load_state_dict(saved["models"][key])
    for key, optimizer in optimizers.items():
        optimizer.load_state_dict(saved["optimizers"][key])
    torch.set_rng_state(saved["torch_rng"])
    rng.bit_generator.state = saved["sampling_rng"]
    return saved

def train_critic(buffer, indices, config, results, checkpoints):
    init_rng(config.seed)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 10]))
    normalization = buffer.observation_stats(indices) if config.normalize_observations else None
    q = QNetwork(buffer.obs_dim, buffer.action_dim, config.hidden_dims, normalization)
    value = Network(buffer.obs_dim, hidden_dims=config.hidden_dims, normalization=normalization)
    auxiliary = Policy(buffer.obs_dim, buffer.action_dim, config.hidden_dims, normalization)
    models = {"q": q, "target_q": deepcopy(q).requires_grad_(False), "value": value,
              "auxiliary": auxiliary, "target_policy": deepcopy(auxiliary).requires_grad_(False)}
    optimizers = {key: torch.optim.Adam(models[key].parameters(), lr=config.learning_rate) for key in ("q", "value", "auxiliary")}
    final, latest = checkpoints / "critic.pt", checkpoints / "critic_latest.pt"
    if final.exists():
        saved = restore(final, models, optimizers, config, rng)

    else:
        start = restore(latest, models, optimizers, config, rng)["step"] if latest.exists() else 0
        trim_losses(results / "critic_losses.csv", start)
        started = time.monotonic()
        with (results / "critic_losses.csv").open("a" if start else "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CRITIC_FIELDS)
            if not start:
                writer.writeheader()
            for step in range(start + 1, config.critic_steps + 1):
                losses = update_critic(models, optimizers, buffer.sample(indices, config.batch_size, rng), config, step)
                if any(not math.isfinite(x) for x in losses.values() if x is not None):
                    raise FloatingPointError(f"Non-finite critic loss at {step}: {losses}")
                writer.writerow({"step": step, **losses})
                if step % 10000 == 0 or step == config.critic_steps:
                    handle.flush()
                    save_state(latest, models, optimizers, config, rng, step, data_info=buffer.info)
                    print(f"critic step={step} updates/s={(step-start)/(time.monotonic()-started):.1f} losses={losses}", flush=True)
        save_state(final, models, optimizers, config, rng, config.critic_steps, data_info=buffer.info)
    for model in models.values():
        model.requires_grad_(False).eval()
    return models

@torch.no_grad()
def cache_weights(models, buffer, config, results):
    path = results / "awr_weights.npy"
    if path.exists() and (results / "awr_weights.json").exists():
        weights = torch.from_numpy(np.load(path))
    else:
        weights = torch.empty(buffer.size)
        for start in range(0, buffer.size, 4096):
            obs = buffer.data["obs"][start:start+4096]
            actions = buffer.data["actions"][start:start+4096]
            advantage = torch.minimum(*models["target_q"](obs, actions)) - models["value"](obs).squeeze(-1)
            weights[start:start+4096] = (config.temperature * advantage).clamp(max=math.log(config.max_weight)).exp()
        temporary = path.with_name(f".awr_weights.{os.getpid()}.npy")
        np.save(temporary, weights.numpy())
        temporary.replace(path)

    precise = weights.double()
    denominator = precise.square().sum().item()
    effective_fraction = precise.sum().item()**2 / denominator / buffer.size if denominator else 0.0
    write_json(results / "awr_weights.json", {"critic_digest": model_digest(models), "mean": weights.mean().item(),
               "min": weights.min().item(), "max": weights.max().item(), "target_q_used": True,
               "zero_fraction": (weights == 0).float().mean().item(),
               "capped_fraction": (weights >= config.max_weight * (1-1e-6)).float().mean().item(),
               "effective_sample_fraction": effective_fraction})
    return weights

@torch.no_grad()
def evaluate_policy(policy, env, seeds):
    was_training = policy.training
    policy.eval()
    episodes = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            episode_return, length = 0.0, 0
            while True:
                action = policy(torch.from_numpy(encode_obs(obs)).unsqueeze(0))[0][0].numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
            episodes.append({"eval_seed": seed, "return": episode_return, "length": length})
    finally:
        policy.train(was_training)
    values = [episode["return"] for episode in episodes]
    return {"mean_return": float(np.mean(values)), "episode_std": float(np.std(values, ddof=1)), "n_episodes": len(episodes)}, episodes

def train_policy(buffer, permutation, models, weights, env, config, fraction, results, checkpoints):
    results.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    final_eval = results / "final_eval.json"
    if final_eval.exists():
        saved = json.loads(final_eval.read_text())
        return saved["summary"]
    indices = permutation[:int(buffer.size * fraction)]
    init_rng(config.seed + 1000)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 20]))
    normalization = buffer.observation_stats(indices) if config.normalize_observations else None
    policy = Policy(buffer.obs_dim, buffer.action_dim, config.hidden_dims, normalization, config.actor_dropout)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    latest = checkpoints / "latest.pt"
    start = restore(latest, {"policy": policy}, {"policy": optimizer}, config, rng)["step"] if latest.exists() else 0
    trim_losses(results / "policy_losses.csv", start)
    curve_path = results / "learning_curve.json"
    curves = [row for row in json.loads(curve_path.read_text()) if row["step"] <= config.critic_steps + start] if start else []
    write_json(results / "config.json", {**settings(config), "policy_fraction": fraction, "policy_transitions": len(indices),
               "policy_init_seed": config.seed + 1000, "critic_checkpoint": str(checkpoints.parent / "critic.pt"),
               "curve_seeds": CURVE_SEEDS, "final_seeds": FINAL_SEEDS,
               "normalizer_fit_transitions": len(indices) if normalization is not None else 0})
    budget = config.total_steps - config.critic_steps
    started = time.monotonic()
    with (results / "policy_losses.csv").open("a" if start else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POLICY_FIELDS)
        if not start:
            writer.writeheader()
        for step in range(start + 1, budget + 1):
            idxs = torch.from_numpy(indices[rng.integers(len(indices), size=config.batch_size)])
            obs, actions = buffer.data["obs"].index_select(0, idxs), buffer.data["actions"].index_select(0, idxs)
            if config.algorithm == "td3_bc":
                loss, bc_loss = td3_policy_loss(models["q"], policy, obs, actions, config.bc_alpha)
            else:
                loss = -(weights.index_select(0, idxs) * gaussian_log_prob(policy, obs, actions)).mean()
                bc_loss = None
            optimizer.param_groups[0]["lr"] = config.learning_rate * 0.5 * (1 + math.cos(math.pi * (step - 1) / budget))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scalar = loss.item()
            if not math.isfinite(scalar):
                raise FloatingPointError(f"Non-finite policy loss at {step}.")
            writer.writerow({"step": step, "policy_loss": scalar, "bc_loss": bc_loss.item() if bc_loss is not None else None})
            total = config.critic_steps + step
            if total % config.eval_every == 0 or step == budget:
                handle.flush()
                summary, episodes = evaluate_policy(policy, env, CURVE_SEEDS)
                curves.append({"step": total, **summary})
                write_json(curve_path, curves)
                write_json(results / f"eval_{total:07d}.json", {"step": total, "summary": summary, "episodes": episodes})
                save_state(latest, {"policy": policy}, {"policy": optimizer}, config, rng, step, policy_fraction=fraction,
                           obs_dim=buffer.obs_dim, action_dim=buffer.action_dim)
                print(f"policy={fraction} step={total} return={summary['mean_return']:.2f} updates/s={(step-start)/(time.monotonic()-started):.1f}", flush=True)
    save_state(checkpoints / "final.pt", {"policy": policy}, {"policy": optimizer}, config, rng, budget,
               policy_fraction=fraction, obs_dim=buffer.obs_dim, action_dim=buffer.action_dim)
    summary, episodes = evaluate_policy(policy, env, FINAL_SEEDS)
    write_json(final_eval, {"step": config.total_steps, "summary": summary, "episodes": episodes})
    return summary

def family_name(config):
    return f"{config.dataset}_{config.algorithm}_c{int(config.critic_fraction*100)}_seed{config.seed}"

def load_policy(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["models"]["policy"]
    normalization = ((state["mean.normalizer.mean"], state["mean.normalizer.std"])
                     if "mean.normalizer.mean" in state else None)
    policy = Policy(saved["obs_dim"], saved["action_dim"], saved["config"]["hidden_dims"],
                    normalization, saved["config"].get("actor_dropout", 0.0))
    policy.load_state_dict(state)
    return policy.eval(), saved

def main(config):
    config = apply_profile(config)
    torch.set_num_threads(1)
    os.environ["MINARI_DATASETS_PATH"] = str(config.data_dir.resolve())
    results, checkpoints = config.results_dir / family_name(config), config.checkpoint_dir / family_name(config)
    results.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    with (results / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = results / "config.json"
        if config_path.exists():
            check_config(json.loads(config_path.read_text()), config)
        write_json(config_path, {**settings(config), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "versions": {name: importlib.metadata.version(name) for name in
                                ("torch", "numpy", "gymnasium", "gymnasium-robotics", "mujoco", "minari")}})
        dataset = minari.load_dataset(DATASETS[config.dataset])
        buffer = OfflineDataset(dataset)
        permutations, manifest = subset_permutations(config, buffer.size)
        indices = permutations["critic"][:int(buffer.size * config.critic_fraction)]
        write_json(results / "dataset.json", {**buffer.info, "critic_transitions": len(indices), "subset_manifest": str(manifest),
                   "critic_normalization": config.normalize_observations,
                   "critic_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(), "subset_rule": "nested random transitions; separate critic/policy permutations"})
        env = dataset.recover_environment(eval_env=True)
        try:
            obs, _ = env.reset(seed=CURVE_SEEDS[0])
            print(f"{family_name(config)} {json.dumps(buffer.info)}", flush=True)
            models = train_critic(buffer, indices, config, results, checkpoints)
            frozen = model_digest(models)
            weights = cache_weights(models, buffer, config, results) if config.algorithm != "td3_bc" else None
            summaries = {}
            for fraction in config.policy_fractions:
                run = f"p{int(fraction*100)}"
                summaries[run] = train_policy(buffer, permutations["policy"], models, weights, env, config, fraction,
                                             results / run, checkpoints / run)
                if model_digest(models) != frozen:
                    raise AssertionError("Critic-stage parameters changed during extraction.")
            write_json(results / "complete.json", {"critic_digest": frozen, "critics_frozen": True,
                       "critic_steps": config.critic_steps, "total_steps_per_policy": config.total_steps, "policies": summaries})
            print("Family complete:", json.dumps(summaries), flush=True)
        finally:
            env.close()

if __name__ == "__main__":
    main(parse_config())
