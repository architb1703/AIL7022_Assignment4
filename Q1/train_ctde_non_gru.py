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

from mapf_env import GAMMA, GRID, FOV, N_AGENTS, T_MAX, SLIP, parallel_env

# Separate seeds for reported evaluation and policy selection.
EVAL_SEEDS = tuple([i for i in range(10000, 10100)])
TRAIN_EVAL_SEEDS = EVAL_SEEDS[:50]
VALID_SEEDS = tuple([i for i in range(20000, 20050)])
RESERVED_SEEDS = frozenset(EVAL_SEEDS + VALID_SEEDS)

# Observation dimensionality
# 4 channel 5x5 window (4x5x5) + Relative dir. (2) + time (1) + Action one-hot (5) + Agent id one-hot (5)
# As distance between agent id's is not indicative of closeness of agent behavior, using one-hot encoding
# for agent ids is a better modelling choice.
OBS_DIM = (4*FOV*FOV) + 2 + 1 + 5 + N_AGENTS

# Global state: walls, all positions, all goals, finished flags and time.
# This input is used only by the mixing network's hypernetworks during training.
STATE_DIM = (GRID*GRID) + (2*N_AGENTS) + (2*N_AGENTS) + N_AGENTS + 1
N_ACTIONS = 5
MAKESPAN_BASELINE = 16.0

# Experiment hyperparameters
@dataclass(frozen=True)
class Config:
    seed: int
    reward_mode: str
    results_dir: Path
    checkpoint_dir: Path

    total_steps: int = int(1e6)
    learning_starts: int = int(5e3)
    train_every: int = 4
    target_update_every: int = int(2e3)
    eval_every: int = int(5e4)

    gamma: float = GAMMA
    learning_rate: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = int(2e4)  # Capacity is measured in joint transitions.
    hidden_dims: tuple[int, int] = (128, 128)
    mixer_hidden_dim: int = 32
    hyper_hidden_dim: int = 64
    max_grad_norm: float = 10.0
    double_q: bool = True
    td_loss: str = "mse"

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = int(5e5)

    device: str = "cpu"

# Separate RNG streams for exploration, replay sampling and training episode seeds.
@dataclass
class RandomStreams:
    exploration: np.random.Generator
    replay: np.random.Generator
    episode_seeds: np.random.Generator

# Load experiment setup details
def parse_config():
    parser = argparse.ArgumentParser(description="Parameter-shared feed-forward QMIX for MAPF.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", choices=("sparse", "shaped"), default="sparse")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--double-q", action=argparse.BooleanOptionalAction, default=Config.double_q)
    parser.add_argument("--td-loss", choices=("huber", "mse"), default=Config.td_loss)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    args = parser.parse_args()

    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32 - 1.")
    if args.seed in RESERVED_SEEDS:
        parser.error("Seeds 10000-10099 and 20000-20049 are reserved for evaluation and validation.")
    if args.total_steps <= 0:
        parser.error("Total training steps must be positive.")
    if not 0 < args.batch_size <= Config.replay_capacity:
        parser.error("Batch size must be positive and no larger than replay capacity.")

    run_name = f"qmix_non_gru_{args.reward_mode}_seed{args.seed}"
    return Config(seed=args.seed, reward_mode=args.reward_mode, total_steps=args.total_steps,
                  results_dir=args.results_dir / run_name, checkpoint_dir=args.checkpoint_dir / run_name, double_q=args.double_q, td_loss=args.td_loss, batch_size=args.batch_size)

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

def encode_state(state):
    state_vector = np.concatenate([
        state["walls"].reshape(-1),
        (state["positions"] / (GRID - 1)).reshape(-1),
        (state["goals"] / (GRID - 1)).reshape(-1),
        state["finished"].astype(np.float32),
        np.array([state["t"] / T_MAX], dtype=np.float32),
    ]).astype(np.float32, copy=False)

    if state_vector.shape != (STATE_DIM,):
        raise ValueError(f"Expected global state shape ({STATE_DIM},), got {state_vector.shape}.")
    return state_vector

# Shared agent utilities. Global state never enters this network.
class QNetwork(nn.Module):
    def __init__(self, inp_dim=OBS_DIM, hidden_dims=(128,128), n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dims[0]), nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], n_actions)
        )

    def forward(self, x):
        if x.shape[-1] != OBS_DIM:
            raise ValueError(f"Agent Q-network expects {OBS_DIM} local features, got {x.shape[-1]}.")
        return self.net(x)

# State-conditioned hypernetworks produce the weights and biases of the mixer.
class QMixer(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_agents=N_AGENTS, mixer_hidden_dim=32, hyper_hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.mixer_hidden_dim = mixer_hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim), nn.ReLU(),
            nn.Linear(hyper_hidden_dim, n_agents*mixer_hidden_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden_dim), nn.ReLU(),
            nn.Linear(hyper_hidden_dim, mixer_hidden_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, mixer_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixer_hidden_dim), nn.ReLU(),
            nn.Linear(mixer_hidden_dim, 1)
        )

    def forward(self, agent_qs, states):
        batch_size = agent_qs.shape[0]

        w1 = self.hyper_w1(states).abs().reshape(batch_size, self.n_agents, self.mixer_hidden_dim)
        b1 = self.hyper_b1(states).reshape(batch_size, 1, self.mixer_hidden_dim)
        hidden = nn.functional.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        w2 = self.hyper_w2(states).abs().reshape(batch_size, self.mixer_hidden_dim, 1)
        b2 = self.hyper_b2(states).reshape(batch_size, 1, 1)
        joint_q = torch.bmm(hidden, w2) + b2
        return joint_q.reshape(batch_size)

class Replay_Buffer:
    def __init__(self, capacity, rng, obs_dim=OBS_DIM, state_dim=STATE_DIM):
        self.capacity = capacity
        self.rng = rng
        self.position, self.size = 0, 0

        self.obs = np.empty((capacity, N_AGENTS, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, N_AGENTS), dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_obs = np.empty((capacity, N_AGENTS, obs_dim), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def update(self, obs, actions, reward, state, next_obs, next_state, done):
        # FIFO style buffer. One entry stores all agents from the same joint step.
        idx = self.position
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = reward
        self.states[idx] = state
        self.next_obs[idx] = next_obs
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = self.rng.choice(self.size, size=batch_size, replace=False)
        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "states": self.states[idxs],
            "next_obs": self.next_obs[idxs],
            "next_states": self.next_states[idxs],
            "dones": self.dones[idxs],
        }

# Exploration Schedule
def get_eps(step, config):
    fraction = min(max(step, 0)/config.epsilon_decay_steps, 1.0)
    eps_range = config.epsilon_end - config.epsilon_start
    return (config.epsilon_start + fraction*eps_range)

@torch.no_grad()
def choose_action(q_network, obs_i, agent_idx, eps, rng, device):
    if eps > 0.0 and rng.random() < eps:
        return int(rng.integers(N_ACTIONS))

    obs_vec = encode_obs(obs_i, agent_idx)
    x = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    q_values = q_network(x)
    return int(q_values.argmax(dim=1).item())

def update_q_network(online_q, target_q, online_mixer, target_mixer, optimizer, buffer, config, device):
    batch = buffer.sample(config.batch_size)
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=device)

    batch_size = obs.shape[0]
    all_q = online_q(obs.reshape(-1, OBS_DIM)).reshape(batch_size, N_AGENTS, N_ACTIONS)
    selected_q = all_q.gather(dim=2, index=actions.unsqueeze(2)).squeeze(2)
    joint_q = online_mixer(selected_q, states)

    with torch.no_grad():
        all_next_q = target_q(next_obs.reshape(-1, OBS_DIM)).reshape(batch_size, N_AGENTS, N_ACTIONS)
        if config.double_q:
            # Online utilities select actions; target utilities evaluate those actions.
            online_next_q = online_q(next_obs.reshape(-1, OBS_DIM)).reshape(batch_size, N_AGENTS, N_ACTIONS)
            next_actions = online_next_q.argmax(dim=2, keepdim=True)
            next_q = all_next_q.gather(dim=2, index=next_actions).squeeze(2)
        else:
            next_q = all_next_q.max(dim=2).values
        next_joint_q = target_mixer(next_q, next_states)
        targets = rewards + config.gamma*(1.0 - dones)*next_joint_q

    loss = nn.functional.mse_loss(joint_q, targets) if config.td_loss == "mse" else nn.functional.smooth_l1_loss(joint_q, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("QMIX TD loss became non-finite.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = list(online_q.parameters()) + list(online_mixer.parameters())
    nn.utils.clip_grad_norm_(parameters, max_norm=config.max_grad_norm)
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
            collisions = 0
            episode_length = 0

            while env.agents:
                actions = {name: choose_action(q_network, obs[name], i, 0.0, eval_rng, device) for i, name in enumerate(agent_names)}
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
    return data

def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)

def save_checkpoint(path, online_q, target_q, online_mixer, target_mixer, optimizer, config, step):
    torch.save({
        "algorithm": "qmix",
        "step": int(step),
        "config": serializable_config(config),
        "online_q": online_q.state_dict(),
        "target_q": target_q.state_dict(),
        "online_mixer": online_mixer.state_dict(),
        "target_mixer": target_mixer.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)

def load_qmix_policy(checkpoint_path, device="cpu"):
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    network = QNetwork(hidden_dims=tuple(checkpoint["config"]["hidden_dims"])).to(device)
    network.load_state_dict(checkpoint["online_q"])
    network.eval()

    def policy_fn(obs_i, agent_idx):
        return choose_action(network, obs_i, agent_idx, 0.0, None, device)

    return policy_fn, checkpoint["config"]

def train(config):
    rngs = init_rng(config.seed)
    device = torch.device(config.device)
    online_q = QNetwork(hidden_dims=config.hidden_dims).to(device)
    target_q = QNetwork(hidden_dims=config.hidden_dims).to(device)
    online_mixer = QMixer(mixer_hidden_dim=config.mixer_hidden_dim, hyper_hidden_dim=config.hyper_hidden_dim).to(device)
    target_mixer = QMixer(mixer_hidden_dim=config.mixer_hidden_dim, hyper_hidden_dim=config.hyper_hidden_dim).to(device)

    target_q.load_state_dict(online_q.state_dict())
    target_mixer.load_state_dict(online_mixer.state_dict())
    target_q.requires_grad_(False)
    target_mixer.requires_grad_(False)
    target_q.eval()
    target_mixer.eval()

    parameters = list(online_q.parameters()) + list(online_mixer.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    buffer = Replay_Buffer(capacity=config.replay_capacity, rng=rngs.replay)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.results_dir / "config.json", serializable_config(config))

    env = parallel_env(reward_mode=config.reward_mode)
    agent_names = list(env.possible_agents)
    step_count = 0
    curve_rows = []

    try:
        obs, _ = env.reset(seed=next_training_seed(rngs.episode_seeds))
        state = encode_state(env.get_global_state())

        with (config.results_dir / "losses.csv").open("w", newline="") as loss_file:
            loss_writer = csv.DictWriter(loss_file, fieldnames=["step", "epsilon", "td_loss"])
            loss_writer.writeheader()

            while step_count < config.total_steps:
                eps = get_eps(step_count, config)
                actions = {name: choose_action(online_q, obs[name], i, eps, rngs.exploration, device) for i, name in enumerate(agent_names)}
                (next_obs,rewards,terminations,truncations,infos) = env.step(actions)
                next_state = encode_state(env.get_global_state())
                episode_done = all(terminations[name] or truncations[name] for name in agent_names)

                buffer.update(
                    obs=np.stack([encode_obs(obs[name], i) for i, name in enumerate(agent_names)]),
                    actions=np.array([actions[name] for name in agent_names], dtype=np.int64),
                    reward=infos[agent_names[0]]["team_reward"],
                    state=state,
                    next_obs=np.stack([encode_obs(next_obs[name], i) for i, name in enumerate(agent_names)]),
                    next_state=next_state,
                    done=episode_done
                )
                step_count += 1

                if (step_count >= config.learning_starts and step_count % config.train_every == 0 and len(buffer) >= config.batch_size):
                    loss = update_q_network(online_q,target_q,online_mixer,target_mixer,optimizer,buffer,config,device)
                    loss_writer.writerow({"step": step_count, "epsilon": eps, "td_loss": loss})

                if step_count % config.target_update_every == 0:
                    target_q.load_state_dict(online_q.state_dict())
                    target_mixer.load_state_dict(online_mixer.state_dict())

                if step_count % config.eval_every == 0:
                    loss_file.flush()
                    summary, episodes = evaluate_policy(online_q,config,device,TRAIN_EVAL_SEEDS)
                    curve_rows.append({"step": step_count, **summary})
                    write_json(config.results_dir / "learning_curve.json", curve_rows)
                    write_json(config.results_dir / f"eval_{step_count:07d}.json", {"step": step_count, "summary": summary, "episodes": episodes})
                    save_checkpoint(config.checkpoint_dir / f"step_{step_count:07d}.pt",online_q,target_q,online_mixer,target_mixer,optimizer,config,step_count)
                    print(f"step={step_count} epsilon={eps:.3f} success={summary['success_rate']:.3f} makespan={summary['makespan_all']:.2f}")

                obs = next_obs
                state = next_state
                if episode_done and step_count < config.total_steps:
                    obs, _ = env.reset(seed=next_training_seed(rngs.episode_seeds))
                    state = encode_state(env.get_global_state())

            save_checkpoint(config.checkpoint_dir / "final.pt",online_q,target_q,online_mixer,target_mixer,optimizer,config,step_count)
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
