from collections import deque
import numpy as np
from mapf_env import T_MAX, parallel_env

def collect_rollout(policy, seed, reward_mode, slip_prob=0.1):
    if hasattr(policy, "reset"):
        policy.reset()
    env = parallel_env(reward_mode=reward_mode, slip_prob=slip_prob)
    obs, _ = env.reset(seed=int(seed))
    names = list(env.possible_agents)
    states, actions, events = [env.get_global_state()], [], []
    collisions = 0
    try:
        while env.agents:
            chosen = {name: policy(obs[name], i) for i, name in enumerate(names)}
            obs, _, _, _, infos = env.step(chosen)
            states.append(env.get_global_state())
            actions.append([chosen[name] for name in names])
            collisions += sum(int(infos[name]["blocked"]) for name in names)
            old, new = states[-2], states[-1]
            moved = (new["positions"] != old["positions"]).any(axis=1)
            for i in range(len(names)):
                if old["finished"][i]:
                    continue
                for j in range(len(names)):
                    if i == j or not moved[j]:
                        continue
                    following = moved[i] and np.array_equal(new["positions"][i], old["positions"][j])
                    waiting = chosen[names[i]] == 0 and np.max(np.abs(old["positions"][i] - old["positions"][j])) <= 2
                    if following or waiting:
                        events.append({"step": len(actions), "agent": i, "other_agent": j, "event": "following" if following else "waiting_near_moving_agent"})
        finish = [int(infos[name]["finish_step"]) for name in names]
        success = bool(infos[names[0]]["success"])
        return {"seed": int(seed), "slip_prob": slip_prob, "success": success, "finish_steps": finish, "makespan": max(finish) if success else T_MAX, "sum_costs": sum(t if t >= 0 else T_MAX for t in finish), "collisions": collisions, "states": states, "actions": actions, "coordination_events": events}
    finally:
        env.close()

def has_path(walls, start, goal, blocked):
    queue, seen = deque([tuple(start)]), {tuple(start)}
    while queue:
        position = queue.popleft()
        if position == tuple(goal):
            return True
        for dr, dc in ((-1,0), (1,0), (0,-1), (0,1)):
            cell = (position[0] + dr, position[1] + dc)
            if 0 <= cell[0] < walls.shape[0] and 0 <= cell[1] < walls.shape[1] and walls[cell] == 0 and cell not in blocked and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return False

def classify_failure(states):
    final = states[-1]
    blocked = {tuple(p) for p in final["positions"][final["finished"]]}
    for i in np.flatnonzero(~final["finished"]):
        if not has_path(final["walls"], final["positions"][i], final["goals"][i], blocked):
            return "goal_blocking"
    stationary, configurations, movements = 0, [], []
    for t, state in enumerate(states):
        unfinished = np.flatnonzero(~state["finished"])
        configurations.append(tuple((int(i), *map(int, state["positions"][i])) for i in unfinished))
        if t == 0:
            continue
        previous = states[t - 1]
        active = ~previous["finished"]
        moved = bool((state["positions"][active] != previous["positions"][active]).any())
        movements.append(moved)
        stationary = 0 if moved else stationary + 1
        if stationary >= 10:
            return "deadlock"
    last_finish = 0
    for t in range(1, len(states)):
        if np.any(states[t]["finished"] != states[t - 1]["finished"]):
            last_finish = t
        if t - last_finish >= 20:
            window = configurations[t - 20:t + 1]
            if len(set(window)) < len(window) and any(movements[t - 20:t]):
                return "livelock"
    return "timeout"
