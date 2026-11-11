"""核对手写梯度、环境奖励和训练闭环，避免教学代码误导。"""

import math
import random
import unittest

from mini_rl import (
    CarrotEnv, collect_episode, discounted_returns, exact_success,
    exact_values, policy_gradient, sigmoid, update,
)


class MiniRLTests(unittest.TestCase):
    def test_returns_and_bootstrap(self):
        for actual, expected in zip(discounted_returns([0, 0, 1], 0.9), [0.81, 0.9, 1]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(discounted_returns([0, 0], 0.9, 0.8)[0], 0.648)

    def test_environment_success_failure_and_reset(self):
        env = CarrotEnv()
        self.assertEqual(env.reset(), 0)
        self.assertEqual(env.step(0), (1, 0.0, False))
        self.assertEqual(env.step(0), (2, 0.0, False))
        self.assertEqual(env.step(0), (None, 1.0, True))
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        self.assertEqual(env.step(1), (None, 0.0, True))

    def test_policy_gradient_matches_finite_difference(self):
        for logit in (-2.0, 0.0, 1.5):
            for action in (0, 1):
                for advantage in (-0.8, 0.0, 0.3):
                    def loss(z):
                        p = sigmoid(z) if action == 0 else sigmoid(-z)
                        return -advantage * math.log(p)
                    eps = 1e-6
                    numerical = (loss(logit + eps) - loss(logit - eps)) / (2 * eps)
                    self.assertAlmostEqual(
                        policy_gradient(logit, action, advantage), numerical, places=8
                    )

    def test_initial_values_and_no_cross_episode_returns(self):
        for actual, expected in zip(exact_values([0, 0, 0], 0.9), [0.10125, 0.225, 0.5]):
            self.assertAlmostEqual(actual, expected)
        logits, values = [0.0] * 3, [0.0] * 3
        success = [{"state": s, "action": 0, "reward": float(s == 2)} for s in range(3)]
        failure = [{"state": 0, "action": 1, "reward": 0.0}]
        update(logits, values, [failure, success], 1.0, 0.0, 1.0)
        # s0 的两个目标是 0 和 1，不能把另一条 episode 的奖励串进失败轨迹。
        self.assertEqual(values, [0.5, 1.0, 1.0])
        self.assertEqual(logits, [0.0] * 3)

    def test_training_improves_actual_policy(self):
        for seed in (7, 19, 42):
            rng = random.Random(seed)
            logits, values = [0.0] * 3, [0.0] * 3
            for _ in range(200):
                batch = [collect_episode(logits, rng) for _ in range(64)]
                update(logits, values, batch, 0.9, 1.0, 0.2)
            self.assertGreater(exact_success(logits), 0.95)
            self.assertTrue(all(math.isfinite(x) for x in logits + values))


if __name__ == "__main__":
    unittest.main()
