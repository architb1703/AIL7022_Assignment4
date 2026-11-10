# Library Imports
import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
import csv
import json

from mapf_env import GAMMA, FOV, N_AGENTS, T_MAX, SLIP, parallel_env

# Separate seeds for reported evaluation and policy selection.
EVAL_SEEDS = tuple([i for i in range(10000, 10100)])
TRAIN_EVAL_SEEDS = EVAL_SEEDS[:50]
VALID_SEEDS = tuple([i for i in range(20000, 20050)])
RESERVED_SEEDS = frozenset(EVAL_SEEDS + VALID_SEEDS)

# Local FOV, goal direction, time, previous action and agent id.
OBS_DIM = (4*FOV*FOV) + 2 + 1 + 5 + N_AGENTS
N_ACTIONS = 5
MAKESPAN_BASELINE = 16.0

# Replay capacity and batch size are measured in complete joint episodes.
@dataclass(frozen=True)
class Config:
    seed: int
    reward_mode: str
    results_dir: Path
    checkpoint_dir: Path

    total_steps: int = int(1e6)
    learning_starts: int = int(5e3)
    train_every_episodes: int = 1
    updates_per_episode: int = 2
    target_update_every: int = int(2e3)
    eval_every: int = int(5e4)

    gamma: float = GAMMA
    learning_rate: float = 5e-4
    batch_size: int = 32
    replay_capacity: int = 1000
    hidden_dim: int = 64
    max_grad_norm: float = 10.0
    double_q: bool = True
    td_loss: str = "mse"

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = int(5e4)

    device: str = "cpu"

@dataclass
class RandomStreams:
    exploration: np.random.Generator
    replay: np.random.Generator
    episode_seeds: np.random.Generator

def parse_config():
    parser = argparse.ArgumentParser(description="Parameter-shared GRU IQL for MAPF.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", choices=("sparse", "shaped"), default="sparse")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--batch-size", type=int, default=32, help="Number of complete episodes per batch.")
    parser.add_argument("--replay-capacity", type=int, default=Config.replay_capacity, help="Maximum number of stored episodes.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--train-every-episodes", type=int, default=1)
    parser.add_argument("--updates-per-episode", type=int, default=Config.updates_per_episode)
    parser.add_argument("--target-update-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50000)
    parser.add_argument("--epsilon-decay-steps", type=int, default=Config.epsilon_decay_steps)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--double-q", action=argparse.BooleanOptionalAction, default=Config.double_q)
    parser.add_argument("--td-loss", choices=("huber", "mse"), default=Config.td_loss)
    args = parser.parse_args()

    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32 - 1.")
    if args.seed in RESERVED_SEEDS:
        parser.error("Seeds 10000-10099 and 20000-20049 are reserved for evaluation and validation.")
    if any(value <= 0 for value in (args.total_steps, args.hidden_dim, args.train_every_episodes, args.target_update_every, args.eval_every, args.epsilon_decay_steps, args.updates_per_episode)):
        parser.error("Step counts, training intervals and hidden dimension must be positive.")
    if args.learning_starts < 0:
        parser.error("Learning starts must be non-negative.")
    if not 0 < args.batch_size <= args.replay_capacity:
        parser.error("Batch size must be positive and no larger than replay capacity.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")

    run_name = f"iql_gru_{args.reward_mode}_seed{args.seed}"
    return Config(seed=args.seed, reward_mode=args.reward_mode, total_steps=args.total_steps,
                  results_dir=args.results_dir / run_name, checkpoint_dir=args.checkpoint_dir / run_name,
                  batch_size=args.batch_size, replay_capacity=args.replay_capacity, hidden_dim=args.hidden_dim,
                  learning_starts=args.learning_starts, train_every_episodes=args.train_every_episodes, updates_per_episode=args.updates_per_episode,
                  target_update_every=args.target_update_every, eval_every=args.eval_every, epsilon_decay_steps=args.epsilon_decay_steps,
                  learning_rate=args.learning_rate, double_q=args.double_q, td_loss=args.td_loss)

def init_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    exploration_seed, replay_seed, episode_seed = np.random.SeedSequence(seed).spawn(3)
    return RandomStreams(exploration=np.random.default_rng(exploration_seed),
                         replay=np.random.default_rng(replay_seed),
                         episode_seeds=np.random.default_rng(episode_seed))

def next_training_seed(rng):
    while True:
        seed = int(rng.integers(0, 2**32))
        if seed not in RESERVED_SEEDS:
            return seed

def encode_obs(obs, agent_idx):
    agent_id = np.zeros(N_AGENTS, dtype=np.float32)
    agent_id[agent_idx] = 1
    obs_vector = np.concatenate([obs["fov"].reshape(-1), obs["delta"], obs["time"], obs["prev_action"], agent_id]).astype(np.float32, copy=False)

    if obs_vector.shape != (OBS_DIM,):
        raise ValueError(f"Expected observation shape ({OBS_DIM},), got {obs_vector.shape}.")
    return obs_vector

# The same GRU is shared by agents, with a separate hidden state for each agent.
class QNetwork(nn.Module):
    def __init__(self, inp_dim=OBS_DIM, hidden_dim=64, n_actions=N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_size):
        return self.fc1.weight.new_zeros((1, batch_size, self.hidden_dim))

    def forward(self, x, hidden=None):
        if x.shape[-1] != OBS_DIM:
            raise ValueError(f"Agent Q-network expects {OBS_DIM} local features, got {x.shape[-1]}.")
        single_step = x.ndim == 2
        if single_step:
            x = x.unsqueeze(1)
        x = nn.functional.relu(self.fc1(x))
        outputs, hidden = self.gru(x, hidden)
        q_values = self.fc2(outputs)
        return (q_values[:, 0] if single_step else q_values), hidden

class Replay_Buffer:
    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.rng = rng
        self.position, self.size = 0, 0
        self.episodes = [None]*capacity

    def __len__(self):
        return self.size

    def update(self, episode):
        # T actions have T+1 observations, including the final bootstrap observation.
        if len(episode["actions"]) == 0 or len(episode["obs"]) != len(episode["actions"]) + 1:
            raise ValueError("An episode must contain T actions and T+1 observations.")
        self.episodes[self.position] = {key: np.array(values, dtype=np.int64 if key == "actions" else np.float32, copy=True) for key, values in episode.items()}
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = self.rng.choice(self.size, size=batch_size, replace=False)
        episodes = [self.episodes[idx] for idx in idxs]
        max_steps = max(len(episode["actions"]) for episode in episodes)
        batch = {}

        for key, values in episodes[0].items():
            sequence_length = max_steps + 1 if key in ("obs", "states") else max_steps
            batch[key] = np.zeros((batch_size, sequence_length, *values.shape[1:]), dtype=values.dtype)
            for idx, episode in enumerate(episodes):
                batch[key][idx, :len(episode[key])] = episode[key]

        batch["mask"] = np.zeros((batch_size, max_steps), dtype=np.float32)
        for idx, episode in enumerate(episodes):
            batch["mask"][idx, :len(episode["actions"])] = 1.0
        return batch

def get_eps(step, config):
    fraction = min(max(step, 0)/config.epsilon_decay_steps, 1.0)
    eps_range = config.epsilon_end - config.epsilon_start
    return (config.epsilon_start + fraction*eps_range)

@torch.no_grad()
def choose_action(q_network, obs_i, agent_idx, hidden, eps, rng, device):
    x = torch.as_tensor(encode_obs(obs_i, agent_idx), dtype=torch.float32, device=device).unsqueeze(0)
    q_values, hidden = q_network(x, hidden)
    action = int(q_values.argmax(dim=1).item())
    # Advance memory even when an exploratory action is selected.
    if eps > 0.0 and rng.random() < eps:
        action = int(rng.integers(N_ACTIONS))
    return action, hidden

@torch.no_grad()
def choose_actions(q_network, obs, agent_names, hidden, eps, rng, device):
    obs_vectors = np.stack([encode_obs(obs[name], i) for i, name in enumerate(agent_names)])
    x = torch.as_tensor(obs_vectors, dtype=torch.float32, device=device)
    q_values, hidden = q_network(x, hidden)
    actions = q_values.argmax(dim=1).tolist()
    for i in range(len(agent_names)):
        if eps > 0.0 and rng.random() < eps:
            actions[i] = int(rng.integers(N_ACTIONS))
    return {name: int(actions[i]) for i, name in enumerate(agent_names)}, hidden

def unroll_q_values(network, obs):
    # Start at the episode boundary for both networks, then align Q_t and Q_(t+1).
    batch_size, sequence_length, n_agents, obs_dim = obs.shape
    sequences = obs.permute(0, 2, 1, 3).reshape(batch_size*n_agents, sequence_length, obs_dim)
    q_values, _ = network(sequences, network.init_hidden(batch_size*n_agents))
    return q_values.reshape(batch_size, n_agents, sequence_length, N_ACTIONS).permute(0, 2, 1, 3)

def update_q_network(online_q, target_q, optimizer, buffer, config, device):
    batch = buffer.sample(config.batch_size)
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(batch["mask"], dtype=torch.float32, device=device).unsqueeze(2)

    all_q = unroll_q_values(online_q, obs)
    selected_q = all_q[:, :-1].gather(dim=3, index=actions.unsqueeze(3)).squeeze(3)

    with torch.no_grad():
        all_next_q = unroll_q_values(target_q, obs)[:, 1:]
        if config.double_q:
            next_actions = all_q[:, 1:].detach().argmax(dim=3, keepdim=True)
            next_q = all_next_q.gather(dim=3, index=next_actions).squeeze(3)
        else:
            next_q = all_next_q.max(dim=3).values
        targets = rewards + config.gamma*(1.0 - dones.unsqueeze(2))*next_q

    td_losses = nn.functional.mse_loss(selected_q, targets, reduction="none") if config.td_loss == "mse" else nn.functional.smooth_l1_loss(selected_q, targets, reduction="none")
    loss = (td_losses*mask).sum() / (mask.sum()*N_AGENTS)
    if not torch.isfinite(loss):
        raise FloatingPointError("IQL GRU TD loss became non-finite.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online_q.parameters(), max_norm=config.max_grad_norm)
    optimizer.step()
    return loss.item()

def evaluate_policy(q_network, config, device, seeds):
    env = parallel_env(reward_mode=config.reward_mode, slip_prob=SLIP)
    agent_names = list(env.possible_agents)
    eval_rng = np.random.default_rng(0)
    was_training = q_network.training
    q_network.eval()
    episodes = []

    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            hidden = q_network.init_hidden(N_AGENTS)
            collisions = 0
            episode_length = 0

            while env.agents:
                actions, hidden = choose_actions(q_network,obs,agent_names,hidden,0.0,eval_rng,device)
                obs, _, _, _, infos = env.step(actions)
                episode_length += 1
                collisions += sum(int(infos[name]["blocked"]) for name in agent_names)

            success = bool(infos[agent_names[0]]["success"])
            finish_steps = [int(infos[name]["finish_step"]) for name in agent_names]
            makespan = max(finish_steps) if success else T_MAX
            sum_costs = sum(finish_step if finish_step >= 0 else T_MAX for finish_step in finish_steps)

            episodes.append({
                "eval_seed": int(seed),
                "success": success,
                "episode_length": episode_length,
                "finish_steps": finish_steps,
                "makespan": makespan,
                "sum_costs": sum_costs,
                "collisions": collisions,
                "normalized_makespan": makespan / MAKESPAN_BASELINE,
            })
    finally:
        env.close()
        q_network.train(was_training)

    successful = [episode for episode in episodes if episode["success"]]
    summary = {
        "n_episodes": len(episodes),
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "makespan_success": (float(np.mean([episode["makespan"] for episode in successful])) if successful else None),
        "makespan_all": float(np.mean([episode["makespan"] for episode in episodes])),
        "sum_costs": float(np.mean([episode["sum_costs"] for episode in episodes])),
        "collisions": float(np.mean([episode["collisions"] for episode in episodes])),
        "normalized_makespan_all": float(np.mean([episode["normalized_makespan"] for episode in episodes])),
    }
    return summary, episodes

def serializable_config(config):
    data = asdict(config)
    data["results_dir"] = str(config.results_dir)
    data["checkpoint_dir"] = str(config.checkpoint_dir)
    data["evaluation_seeds"] = list(EVAL_SEEDS)
    data["learning_curve_eval_seeds"] = list(TRAIN_EVAL_SEEDS)
    data["validation_seeds"] = list(VALID_SEEDS)
    data["replay_unit"] = "complete_joint_episodes"
    return data

def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)

def save_checkpoint(path, online_q, target_q, optimizer, config, step, episode_count):
    torch.save({
        "algorithm": "iql_gru",
        "step": int(step),
        "episodes": int(episode_count),
        "config": serializable_config(config),
        "online_q": online_q.state_dict(),
        "target_q": target_q.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)

def load_iql_policy(checkpoint_path, device="cpu"):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    network = QNetwork(hidden_dim=checkpoint["config"]["hidden_dim"]).to(device)
    network.load_state_dict(checkpoint["online_q"])
    network.eval()
    hidden_states = {}

    def policy_fn(obs_i, agent_idx):
        # Time-zero observations also reset memory when using the existing GIF helper.
        hidden = None if float(obs_i["time"][0]) == 0.0 else hidden_states.get(agent_idx)
        action, hidden = choose_action(network,obs_i,agent_idx,hidden,0.0,None,device)
        hidden_states[agent_idx] = hidden
        return action

    policy_fn.reset = hidden_states.clear
    return policy_fn, checkpoint["config"]

def train(config):
    rngs = init_rng(config.seed)
    device = torch.device(config.device)
    online_q = QNetwork(hidden_dim=config.hidden_dim).to(device)
    target_q = QNetwork(hidden_dim=config.hidden_dim).to(device)
    target_q.load_state_dict(online_q.state_dict())
    target_q.requires_grad_(False)
    target_q.eval()

    optimizer = torch.optim.Adam(online_q.parameters(), lr=config.learning_rate)
    buffer = Replay_Buffer(capacity=config.replay_capacity, rng=rngs.replay)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.results_dir / "config.json", serializable_config(config))

    env = parallel_env(reward_mode=config.reward_mode)
    agent_names = list(env.possible_agents)
    step_count, episode_count = 0, 0
    curve_rows = []

    try:
        obs, _ = env.reset(seed=next_training_seed(rngs.episode_seeds))
        hidden = online_q.init_hidden(N_AGENTS)
        episode = {"obs": [np.stack([encode_obs(obs[name], i) for i, name in enumerate(agent_names)])], "actions": [], "rewards": [], "dones": []}

        with (config.results_dir / "losses.csv").open("w", newline="") as loss_file:
            loss_writer = csv.DictWriter(loss_file, fieldnames=["step", "episode", "epsilon", "td_loss"])
            loss_writer.writeheader()

            while step_count < config.total_steps:
                eps = get_eps(step_count, config)
                actions, hidden = choose_actions(online_q,obs,agent_names,hidden,eps,rngs.exploration,device)
                (next_obs,rewards,terminations,truncations,infos) = env.step(actions)
                episode_done = all(terminations[name] or truncations[name] for name in agent_names)

                episode["obs"].append(np.stack([encode_obs(next_obs[name], i) for i, name in enumerate(agent_names)]))
                episode["actions"].append([actions[name] for name in agent_names])
                episode["rewards"].append([rewards[name] + infos[name]["team_bonus"] for name in agent_names])
                episode["dones"].append(float(episode_done))
                step_count += 1

                if episode_done:
                    episode_count += 1
                    buffer.update(episode)
                    if (step_count >= config.learning_starts and episode_count % config.train_every_episodes == 0 and len(buffer) >= config.batch_size):
                        for _ in range(config.updates_per_episode):
                            loss = update_q_network(online_q,target_q,optimizer,buffer,config,device)
                            loss_writer.writerow({"step": step_count, "episode": episode_count, "epsilon": eps, "td_loss": loss})

                if step_count % config.target_update_every == 0:
                    target_q.load_state_dict(online_q.state_dict())

                if step_count % config.eval_every == 0:
                    loss_file.flush()
                    summary, episodes = evaluate_policy(online_q,config,device,TRAIN_EVAL_SEEDS)
                    curve_rows.append({"step": step_count, **summary})
                    write_json(config.results_dir / "learning_curve.json", curve_rows)
                    write_json(config.results_dir / f"eval_{step_count:07d}.json", {"step": step_count, "summary": summary, "episodes": episodes})
                    save_checkpoint(config.checkpoint_dir / f"step_{step_count:07d}.pt",online_q,target_q,optimizer,config,step_count,episode_count)
                    print(f"step={step_count} episodes={episode_count} epsilon={eps:.3f} success={summary['success_rate']:.3f} makespan={summary['makespan_all']:.2f}")

                obs = next_obs
                if episode_done and step_count < config.total_steps:
                    obs, _ = env.reset(seed=next_training_seed(rngs.episode_seeds))
                    hidden = online_q.init_hidden(N_AGENTS)
                    episode = {"obs": [np.stack([encode_obs(obs[name], i) for i, name in enumerate(agent_names)])], "actions": [], "rewards": [], "dones": []}

            # A final unfinished episode is excluded from replay; the step budget stays exact.
            save_checkpoint(config.checkpoint_dir / "final.pt",online_q,target_q,optimizer,config,step_count,episode_count)
            write_json(config.results_dir / "learning_curve.json", curve_rows)

            validation_summary, validation_episodes = evaluate_policy(online_q,config,device,VALID_SEEDS)
            write_json(config.results_dir / "validation_eval.json", {"step": step_count, "summary": validation_summary, "episodes": validation_episodes})
            print("Validation evaluation:")
            print(json.dumps(validation_summary, indent=2))

            final_summary, final_episodes = evaluate_policy(online_q,config,device,EVAL_SEEDS)
            write_json(config.results_dir / "final_eval.json", {"step": step_count, "summary": final_summary, "episodes": final_episodes})
            print("Final evaluation:")
            print(json.dumps(final_summary, indent=2))
            return final_summary
    finally:
        env.close()

if __name__ == "__main__":
    config = parse_config()
    train(config)
