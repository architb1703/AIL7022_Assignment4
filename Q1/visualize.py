"""Generates env_visualization.png (static map) and a GIF of one rollout.

    python make_visualizations.py                 # random-policy placeholder rollout
    from make_visualizations import save_rollout_gif
    save_rollout_gif(policy_fn, "final_policy.gif")   # policy_fn(obs_dict_of_one_agent, agent_idx) -> action
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from PIL import Image
from mapf_env import CustomMAPFParallelEnv


def save_static(path="env_visualization.png"):
    env = CustomMAPFParallelEnv(render_mode="rgb_array")
    env.reset(seed=0)
    env.render(show_fov=True)
    env.fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    env.close()
    print("saved", path)


def rollout_frames(policy_fn, seed=10000, slip_prob=0.1, reward_mode="sparse", max_frames=None):
    """Greedy decentralized rollout. policy_fn sees ONLY one agent's own observation."""
    env = CustomMAPFParallelEnv(render_mode="rgb_array", slip_prob=slip_prob, reward_mode=reward_mode)
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    while env.agents:
        acts = {n: policy_fn(obs[n], i) for i, n in enumerate(env.possible_agents) if n in env.agents}
        obs, _, _, _, info = env.step(acts)
        frames.append(env.render())
        if max_frames and len(frames) >= max_frames:
            break
    env.close()
    return frames


def save_gif(frames, path, fps=3):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    print("saved", path, f"({len(frames)} frames)")


def save_rollout_gif(policy_fn, path="final_policy.gif", **kw):
    save_gif(rollout_frames(policy_fn, **kw), path)


if __name__ == "__main__":
    save_static("env_visualization.png")
    rng = np.random.default_rng(0)
    save_rollout_gif(lambda o, i: int(rng.integers(5)), "random_rollout.gif", max_frames=30)