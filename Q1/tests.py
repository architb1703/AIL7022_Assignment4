import unittest
import numpy as np
from mapf_env import parallel_env, STAY, UP, DOWN, LEFT, RIGHT

class TestCustomMAPFParallelEnv(unittest.TestCase):

    def make_env(self, shape, starts, goals, walls, slip_prob=0.0, **kwargs):
        grid = np.zeros(shape, dtype=np.int8)
        for (x,y) in walls:
            grid[x,y] = 1
        layout = (grid, starts, goals)
        kwargs['layout'] = layout
        kwargs['slip_prob'] = slip_prob
        env = parallel_env(**kwargs)
        env.reset(seed=42)
        return env

    def step(self, env, actions):
        self.assertEqual(len(actions), len(env.possible_agents))
        result = env.step(dict(zip(env.possible_agents, actions)))
        return result

    def assert_pos(self, env, expected):
        np.testing.assert_array_equal(env.pos, expected)

    def assert_reward(self, result, expected):
        np.testing.assert_allclose(list(result[1].values()), expected, atol=1e-8)

    def assert_blocking(self, result, expected):
        self.assertEqual([info["blocked"] for info in result[4].values()], expected)

    def assert_nested_equal(self, actual, expected):
        if isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(actual, expected)

    def test_wall(self):
        # (i) Moving into a wall should result in agent staying in place, 
        # receiving a reward of -1.05, and being flagged as blocked.
        env = self.make_env((2,2), [(0,0)], [(1,1)], [(0,1)])
        action = [RIGHT]
        result = self.step(env, action)
        self.assert_pos(env, [(0,0)])
        self.assert_reward(result, [-1.05])
        self.assert_blocking(result, [True])

    # Additional Test
    def test_boundary(self):
        # Moving into a boundary should result in agent staying in place, 
        # receiving a reward of -1.05, and being flagged as blocked.
        env = self.make_env((2,2), [(0,0)], [(1,1)], [(0,1)])
        action = [LEFT]
        result = self.step(env, action)
        self.assert_pos(env, [(0,0)])
        self.assert_reward(result, [-1.05])
        self.assert_blocking(result, [True])

    def test_vertex_conflict(self):
        # (ii) Two agents trying to enter the same empty cell should both be blocked
        # and penalized with reward -1.05
        env = self.make_env((3,3), [(0,0),(0,2)], [(1,1),(1,2)], [])
        action = [RIGHT, LEFT]
        result = self.step(env, action)
        self.assert_pos(env, [(0,0), (0,2)])
        self.assert_reward(result, [-1.05, -1.05])
        self.assert_blocking(result, [True, True])

    def test_swap_conflict(self):
        # (iii) Agent i trying to move into agent j's current cell, and vice versa.
        # Both agents should be blocked and penalized.
        env = self.make_env((3,3), [(0,0),(0,1)], [(1,1),(1,2)], [])
        action = [RIGHT, LEFT]
        result = self.step(env, action)
        self.assert_pos(env, [(0,0), (0,1)])
        self.assert_reward(result, [-1.05, -1.05])
        self.assert_blocking(result, [True, True])

    def test_following(self):
        # (iv) Agent i can enter a cell that agent j is leaving provided agent j is not 
        # moving into agent i's current cell.
        env = self.make_env((3,3), [(0,0),(0,1)], [(1,1),(1,2)], [])
        action = [RIGHT, RIGHT]
        result = self.step(env, action)
        self.assert_pos(env, [(0,1), (0,2)])
        self.assert_reward(result, [-0.05, -0.05])
        self.assert_blocking(result, [False, False])

    def test_cascade(self):
        # (v) The following movement is blocked as the block cascades through the chain.
        env = self.make_env((3,3), [(0,0),(0,1),(0,2)], [(1,1),(1,2), (2,2)], [])
        action = [RIGHT, RIGHT, RIGHT]
        result = self.step(env, action)
        self.assert_pos(env, [(0,0),(0,1),(0,2)])
        self.assert_reward(result, [-1.05, -1.05, -1.05])
        self.assert_blocking(result, [True, True, True])

    def test_rotation(self):
        # (vi) For agents positioned in a 2x2 block can all move to a new position in the block, 
        # if all the movements are along the same rotation around the sqaure.
        env = self.make_env((3,3), [(0,0),(0,1),(1,1),(1,0)], [(1,1),(1,2), (2,2), (0,2)], [])
        action = [RIGHT, DOWN, LEFT, UP]
        result = self.step(env, action)
        self.assert_pos(env, [(0,1),(1,1),(1,0),(0,0)])
        self.assert_reward(result, [-0.05, -0.05, -0.05, -0.05])
        self.assert_blocking(result, [False,False,False,False])        
    
    def test_finished(self):
        # (vii) If agent i tries to move into the cell of a finished agent j, 
        # then i is penalized and blocked. Agent j does not face any affect.
        env = self.make_env((3,3), [(0,0),(0,1)], [(0,1),(2,2)], [])
        action = [RIGHT, DOWN]
        result = self.step(env, action)
        self.assert_pos(env, [(0,1), (1,1)])
        self.assert_reward(result, [10, -0.05])

        action = [LEFT, UP]
        result = self.step(env, action)
        self.assert_pos(env, [(0,1), (1,1)])
        self.assert_reward(result, [0, -1.05])
        self.assert_blocking(result, [False,True])   

    def test_all_slip(self):
        # (viii) If slip probability is 1, then all agents will slip and stay at same position.
        # No one will be penalized by the blocking penalty.
        env = self.make_env((3,3), [(0,0),(0,1)], [(0,1),(2,2)], [], slip_prob=1.0)
        action = [RIGHT, DOWN]
        result = self.step(env, action)
        self.assert_pos(env, [(0,0), (0,1)])
        self.assert_reward(result, [-0.05, -0.05])
        self.assert_blocking(result, [False,False])   

    def test_sparse_rewards(self):
        # (ix) Testing different per-agent rewards and team level bonuses in an episode.
        env = self.make_env((3,3), [(0,0),(0,1)], [(0,2),(1,2)], [])

        # (a) Both agent move unresticted to non-goal cells. Both should receive -0.05 reward and 
        # team bonus should be 0.
        action = [RIGHT, RIGHT]
        result = self.step(env, action)
        self.assert_pos(env, [(0,1), (0,2)])
        self.assert_reward(result, [-0.05, -0.05])
        for info in result[4].values():
            self.assertEqual(info["team_bonus"], 0)
            self.assertAlmostEqual(info["team_reward"], -0.1)

        # (b) Agent 0 makes a blocked move and receives additional penalty. (total -1.05)
        # Agent 1 moves into goal cell and gets reward of 10. Team bonus will be 0.
        action = [UP, DOWN]
        result = self.step(env, action)
        self.assert_pos(env, [(0,1), (1,2)])
        self.assert_reward(result, [-1.05, 10])
        for info in result[4].values():
            self.assertEqual(info["team_bonus"], 0)
            self.assertAlmostEqual(info["team_reward"], 8.95)

        # (c) Agent 0 moves into goal cell. Agent 1 gets no additional reward after finishing.
        # Team bonus will be 5.
        action = [RIGHT, UP]
        result = self.step(env, action)
        self.assert_pos(env, [(0,2), (1,2)])
        self.assert_reward(result, [10, 0])
        for info in result[4].values():
            self.assertEqual(info["team_bonus"], 5)
            self.assertAlmostEqual(info["team_reward"], 15)

    def test_shaped_reward(self):
        # (ix) Shaped rewards equal sparse rewards plus 0.1(0.99 Phi(new)-Phi(old)).
        orig = self.make_env((3,3), [(0,0)], [(0,2)], [])
        shaped = self.make_env((3,3), [(0,0)], [(0,2)], [], reward_mode="shaped")

        cases = [(UP, -1.05, -2, -2), (RIGHT, -0.05, -2, -1), (RIGHT, 10, -1, 0)]
        for action, base_reward, phi_old, phi_new in cases:
            orig_result = self.step(orig, [action])
            shaped_result = self.step(shaped, [action])
            expected_reward = base_reward + 0.1*(0.99*phi_new - phi_old)
            self.assert_reward(orig_result, [base_reward])
            self.assert_reward(shaped_result, [expected_reward])

    def test_equal_seeds_and_actions_are_deterministic(self):
        # (x) Equal reset seeds and joint actions reproduce all outputs despite slip noise.
        first = self.make_env((3,3), [(0,0)], [(0,2)], [])
        second = self.make_env((3,3), [(0,0)], [(0,2)], [])
        self.assert_nested_equal(first.reset(seed=42), second.reset(seed=42))
        rng = np.random.default_rng(42)
        while first.agents:
            actions = [int(rng.integers(5)) for _ in first.possible_agents]
            self.assert_nested_equal(self.step(first, actions), self.step(second, actions))
            self.assert_nested_equal(first.get_global_state(), second.get_global_state())

if __name__ == "__main__":
    unittest.main()