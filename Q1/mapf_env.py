"""Cooperative decentralized MAPF as a Dec-POMDP (PettingZoo ParallelEnv).

Follows the Assignment-4 specification exactly:
  slip -> conflict resolution (wall / vertex / swap, cascades to a fixed point,
  following and rotation allowed) -> absorbing goals -> rewards -> observations.

Reward bookkeeping
  * rewards[agent]            : r_i^t (sparse) or r_i^t + shaping (shaped). NO team bonus.
  * infos[agent]["team_bonus"]: R_team (=5) on the step where all agents finish, else 0.
  * infos[agent]["team_reward"]: r^t = sum_i r_i^t (+shaping) + R_team  (use for CTDE).
  * Independent learners: r_i^t + infos[agent]["team_bonus"].
"""
import functools
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import ParallelEnv

# ---- constants (do not change) -------------------------------------------
N_AGENTS, GRID, FOV, T_MAX = 5, 11, 5, 80
SLIP, GAMMA = 0.1, 0.99
R_GOAL, C_STEP, C_COL, R_TEAM, BETA = 10.0, -0.05, -1.0, 5.0, 0.1
STAY, UP, DOWN, LEFT, RIGHT = range(5)
DR = np.array([0, -1, 1, 0, 0])
DC = np.array([0, 0, 0, -1, 1])

MAP_ROWS = [
    "...##.##...",
    "...##.##...",
    "...##.##...",
    "##.##.##.##",
    "##.##.##.##",
    "...........",
    "##.##.##.##",
    "##.##.##.##",
    "...##.##...",
    "...##.##...",
    "...##.##...",
]
STARTS = [(1, 1), (9, 1), (5, 0), (0, 5), (3, 8)]
GOALS = [(9, 9), (1, 9), (5, 10), (10, 5), (5, 1)]


def default_layout():
    grid = np.array([[1 if ch == "#" else 0 for ch in row] for row in MAP_ROWS], dtype=np.int8)
    return grid, list(STARTS), list(GOALS)


def bfs_dist(grid, goal):
    """Shortest-path length to `goal` from every cell, ignoring agents (inf for walls)."""
    H, W = grid.shape
    d = np.full((H, W), np.inf)
    d[goal] = 0
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for a in range(1, 5):
            nr, nc = r + DR[a], c + DC[a]
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0 and d[nr, nc] == np.inf:
                d[nr, nc] = d[r, c] + 1
                q.append((nr, nc))
    return d


def parallel_env(**kwargs):
    return CustomMAPFParallelEnv(**kwargs)


class CustomMAPFParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "mapf_dec_pomdp_v1"}

    COLORS = ["#D62728", "#1F77B4", "#2CA02C", "#F2B600", "#8E44AD"]

    def __init__(self, fov_size=FOV, max_steps=T_MAX, slip_prob=SLIP, reward_mode="sparse",
                 gamma=GAMMA, beta=BETA, layout=None, render_mode=None):
        """`layout=(grid, starts, goals)` is only meant for unit tests; the default is the fixed map."""
        super().__init__()
        assert reward_mode in ("sparse", "shaped")
        grid, starts, goals = default_layout() if layout is None else layout
        self.grid = np.asarray(grid, dtype=np.int8)
        self.H, self.W = self.grid.shape
        self.starts = np.array(starts, dtype=int)
        self.goals = np.array(goals, dtype=int)
        self.N = len(starts)
        self.k, self.T = fov_size, max_steps
        self.slip_prob, self.reward_mode = slip_prob, reward_mode
        self.gamma, self.beta = gamma, beta
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(self.N)]
        self.agents = []
        self.dist = np.stack([bfs_dist(self.grid, tuple(g)) for g in self.goals])  # (N,H,W)
        self.rng = np.random.default_rng()
        self.fig = self.ax = None

    # ---- spaces -----------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Dict({
            "fov": Box(0.0, 1.0, (4, self.k, self.k), np.float32),
            "delta": Box(-1.0, 1.0, (2,), np.float32),
            "time": Box(0.0, 1.0, (1,), np.float32),
            "prev_action": Box(0.0, 1.0, (5,), np.float32),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)

    # ---- helpers ----------------------------------------------------------
    def _d(self, i, p):
        return self.dist[i, p[0], p[1]]

    def _obs(self, i):
        r, c = self.pos[i]
        h = self.k // 2
        fov = np.zeros((4, self.k, self.k), np.float32)
        for kr in range(-h, h + 1):
            for kc in range(-h, h + 1):
                gr, gc, tr, tc = r + kr, c + kc, kr + h, kc + h
                if not (0 <= gr < self.H and 0 <= gc < self.W):
                    fov[0, tr, tc] = 1.0
                    continue
                fov[0, tr, tc] = self.grid[gr, gc]
                for j in range(self.N):
                    if j == i:
                        continue
                    if self.pos[j][0] == gr and self.pos[j][1] == gc:
                        fov[1, tr, tc] = 1.0
                    if self.goals[j][0] == gr and self.goals[j][1] == gc:
                        fov[2, tr, tc] = 1.0
                if self.goals[i][0] == gr and self.goals[i][1] == gc:
                    fov[3, tr, tc] = 1.0
        return {
            "fov": fov,
            "delta": ((self.goals[i] - self.pos[i]) / max(self.H, self.W)).astype(np.float32),
            "time": np.array([self.t / self.T], np.float32),
            "prev_action": self.prev_action[i].copy(),
        }

    def _all_obs(self):
        return {n: self._obs(i) for i, n in enumerate(self.possible_agents)}

    def get_global_state(self):
        """Centralized training only (critic / mixing network). Never feed to actors / Q-nets."""
        return {
            "walls": self.grid.copy(),
            "positions": self.pos.copy(),
            "goals": self.goals.copy(),
            "finished": self.finished.copy(),
            "t": self.t,
        }

    # ---- API --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.pos = self.starts.copy()
        self.finished = np.zeros(self.N, bool)
        self.finish_step = np.full(self.N, -1)
        self.prev_action = np.zeros((self.N, 5), np.float32)
        self.t = 0
        return self._all_obs(), {n: {} for n in self.agents}

    def step(self, actions):
        N = self.N
        self.t += 1
        pos = self.pos.copy()
        was_finished = self.finished.copy()
        active = ~was_finished

        # (1) chosen actions, (2) slip (an independent draw for every unfinished agent)
        chosen = np.zeros(N, int)
        act = np.zeros(N, int)
        for i, name in enumerate(self.possible_agents):
            if was_finished[i]:
                continue
            chosen[i] = act[i] = int(actions.get(name, STAY))
            if self.rng.random() < self.slip_prob:
                act[i] = STAY
        moving = (act != STAY) & active
        target = pos + np.stack([DR[act], DC[act]], axis=1)

        # (3) conflict resolution (blocked set only grows -> fixed point)
        blocked = np.zeros(N, bool)
        for i in np.flatnonzero(moving):
            r, c = target[i]
            if not (0 <= r < self.H and 0 <= c < self.W) or self.grid[r, c] == 1:
                blocked[i] = True
        changed = True
        while changed:
            changed = False
            go = moving & ~blocked
            eff = np.where(go[:, None], target, pos)  # where each agent ends up if nothing else changes
            for i in np.flatnonzero(go):
                for j in range(N):
                    if j == i:
                        continue
                    vertex = (eff[j] == target[i]).all()  # j also enters it, or j stays in it
                    swap = go[j] and (target[j] == pos[i]).all() and (target[i] == pos[j]).all()
                    if vertex or swap:
                        blocked[i] = True
                        changed = True
                        break
        executed = moving & ~blocked
        new_pos = np.where(executed[:, None], target, pos)
        events = moving & blocked  # blocked-move events (post-slip move that did not happen)

        # (4) rewards, finishing, observations
        reached = active & (new_pos == self.goals).all(axis=1)
        r = R_GOAL * reached + C_STEP * (active & ~reached) + C_COL * events
        if self.reward_mode == "shaped":
            for i in np.flatnonzero(active):
                phi_old, phi_new = -self._d(i, pos[i]), -self._d(i, new_pos[i])
                r[i] += self.beta * (self.gamma * phi_new - phi_old)
        self.pos = new_pos
        self.finished = was_finished | reached
        self.finish_step[reached] = self.t
        self.prev_action = np.zeros((N, 5), np.float32)
        for i in np.flatnonzero(active):
            self.prev_action[i, chosen[i]] = 1.0

        success = bool(self.finished.all())
        truncated = (self.t >= self.T) and not success
        bonus = R_TEAM if success else 0.0
        team_reward = float(r.sum() + bonus)

        names = self.possible_agents
        obs = self._all_obs()
        rewards = {n: float(r[i]) for i, n in enumerate(names)}
        terminations = {n: success for n in names}
        truncations = {n: truncated for n in names}
        infos = {n: {"blocked": bool(events[i]), "finish_step": int(self.finish_step[i]),
                     "team_bonus": bonus, "team_reward": team_reward, "success": success}
                 for i, n in enumerate(names)}
        if success or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    # ---- rendering --------------------------------------------------------
    def render(self, show_fov=True):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax = self.ax
        ax.clear()
        ax.set_xlim(-0.5, self.W - 0.5)
        ax.set_ylim(self.H - 0.5, -0.5)  # row 0 on top
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(-0.5, self.W, 1))
        ax.set_yticks(np.arange(-0.5, self.H, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color="#D3D3D3", linewidth=1.0)
        ax.tick_params(length=0)
        for r in range(self.H):
            for c in range(self.W):
                if self.grid[r, c]:
                    ax.add_patch(patches.Rectangle((c - .5, r - .5), 1, 1, fc="#2C3E50", ec="none", zorder=1))
        h = self.k // 2
        pos = getattr(self, "pos", self.starts)
        fin = getattr(self, "finished", np.zeros(self.N, bool))
        for i in range(self.N):
            col = self.COLORS[i]
            if show_fov and not fin[i]:
                xy = (pos[i][1] - h - .5, pos[i][0] - h - .5)
                ax.add_patch(patches.Rectangle(xy, self.k, self.k, fc=col, alpha=0.12, ec="none", zorder=0))
                ax.add_patch(patches.Rectangle(xy, self.k, self.k, fc="none", ec=col, lw=1.2, ls=":",
                                               alpha=0.8, zorder=4))
        for i in range(self.N):  # goals: dashed outline with the agent index
            col = self.COLORS[i]
            gr, gc = self.goals[i]
            ax.add_patch(patches.Rectangle((gc - .4, gr - .4), .8, .8, fc="none", ec=col, lw=2, ls="--", zorder=5))
            ax.text(gc, gr, str(i), color=col, weight="bold", fontsize=15, ha="center", va="center", zorder=6)
        for i in range(self.N):  # agents: filled square with the agent index
            col = self.COLORS[i]
            ax.add_patch(patches.FancyBboxPatch((pos[i][1] - .34, pos[i][0] - .34), .68, .68,
                                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                                fc=col, ec="white", lw=1.5, alpha=0.45 if fin[i] else 1.0, zorder=7))
            ax.text(pos[i][1], pos[i][0], str(i), color="black" if i == 3 else "white",
                    weight="bold", fontsize=15, ha="center", va="center", zorder=8)
        ax.text(0.02, 1.015, f"Step: {getattr(self, 't', 0)}", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom")
        if self.render_mode == "human":
            plt.pause(0.1)
        else:
            self.fig.canvas.draw()
            return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = self.ax = None