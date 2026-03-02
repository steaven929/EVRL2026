import unittest
import numpy as np

from core.algorithm import TD3
from core.buffer import ReplayBuffer
from specific.model import Actor, Critic


class TD3IntegrationTest(unittest.TestCase):
    def test_td3_update_returns_twin_critic_losses(self):
        n_states = 10
        n_actions = 12
        cfg = {
            "device": "cpu",
            "batch_size": 8,
            "gamma": 0.97,
            "tau": 3e-4,
            "actor_lr": 2e-5,
            "critic_lr": 3e-5,
            "policy_delay": 2,
            "target_policy_noise": 0.1,
            "target_noise_clip": 0.2,
        }

        models = {
            "actor": Actor(n_states, n_actions, embedding_dims=8, hidden_dim=64),
            "critic": Critic(n_states, n_actions, embedding_dims=8, hidden_dim=64),
            "critic2": Critic(n_states, n_actions, embedding_dims=8, hidden_dim=64),
        }
        memories = {"memory": ReplayBuffer(256)}

        agent = TD3(models, memories, cfg)

        for _ in range(64):
            state = np.random.randint(0, 5, size=(n_states,), dtype=np.int64)
            action = np.random.uniform(0.0, 1.0, size=(n_actions,)).astype(np.float32)
            reward = float(np.random.randn())
            next_state = np.random.randint(0, 5, size=(n_states,), dtype=np.int64)
            done = bool(np.random.rand() > 0.7)
            agent.memory.push((state, action, reward, next_state, done))

        loss_info = agent.update()
        self.assertIsInstance(loss_info, dict)
        self.assertIn("critic1_loss", loss_info)
        self.assertIn("critic2_loss", loss_info)


if __name__ == "__main__":
    unittest.main()
