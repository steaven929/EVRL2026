import unittest

import numpy as np
import torch

from core.marl_training import compute_exploration_noise
from specific.marl_model import CentralizedCritic, IndependentActorEnsemble, SharedActor


class AblationSwitchTests(unittest.TestCase):
    def test_disabling_zone_embedding_removes_agent_id_effect(self):
        torch.manual_seed(7)
        actor = SharedActor(
            obs_dim=2, n_agents=2, agent_embed_dim=3,
            hidden_dim=8, action_dim=3, use_zone_embedding=False,
        )
        obs = torch.tensor([[0.2, 0.4], [0.2, 0.4]])
        result = actor(obs, torch.tensor([0, 1]))
        torch.testing.assert_close(result[0], result[1])

    def test_zone_ablation_preserves_historical_actor_input_dimension(self):
        for enabled in (True, False):
            actor = SharedActor(
                obs_dim=6, n_agents=50, agent_embed_dim=16,
                hidden_dim=128, action_dim=3, use_zone_embedding=enabled,
            )
            self.assertEqual(actor.fc1.in_features, 22)

    def test_disabling_global_context_masks_critic_input_without_dimension_change(self):
        torch.manual_seed(8)
        critic = CentralizedCritic(
            global_state_dim=4, n_agents=2, action_dim=3,
            hidden_dim=8, use_global_context=False,
        )
        actions = torch.ones((2, 6))
        result = critic(torch.tensor([[0.0] * 4, [9.0] * 4]), actions)
        torch.testing.assert_close(result[0], result[1])
        self.assertEqual(critic.fc1.in_features, 10)

    def test_independent_actor_ensemble_owns_distinct_actor_parameters(self):
        ensemble = IndependentActorEnsemble(
            obs_dim=2, n_agents=2, agent_embed_dim=3,
            hidden_dim=8, action_dim=3, use_zone_embedding=True,
        )
        self.assertEqual(len(ensemble.actors), 2)
        first = next(ensemble.actors[0].parameters())
        second = next(ensemble.actors[1].parameters())
        self.assertNotEqual(first.data_ptr(), second.data_ptr())


class AdaptiveExplorationTests(unittest.TestCase):
    def test_adaptive_noise_increases_below_service_target(self):
        base = compute_exploration_noise(
            episode=50,
            train_episodes=100,
            schedule="linear",
            noise_start=0.12,
            noise_min=0.02,
            decay_power=1.0,
        )
        adaptive = compute_exploration_noise(
            episode=50,
            train_episodes=100,
            schedule="adaptive",
            noise_start=0.12,
            noise_min=0.02,
            decay_power=1.0,
            service_rate_ema=0.60,
            service_target=0.85,
            service_boost=0.05,
        )
        self.assertGreater(adaptive.final_sigma, base.final_sigma)
        self.assertAlmostEqual(adaptive.service_gap, 0.25)

    def test_adaptive_noise_equals_base_at_or_above_target(self):
        value = compute_exploration_noise(
            episode=10,
            train_episodes=100,
            schedule="adaptive",
            noise_start=0.12,
            noise_min=0.02,
            decay_power=2.0,
            service_rate_ema=0.90,
            service_target=0.85,
            service_boost=0.05,
        )
        self.assertEqual(value.service_gap, 0.0)
        self.assertAlmostEqual(value.final_sigma, value.base_sigma)

    def test_noise_schedules_respect_configured_bounds(self):
        for schedule in ("adaptive", "fixed", "linear", "exponential"):
            value = compute_exploration_noise(
                episode=1000,
                train_episodes=100,
                schedule=schedule,
                noise_start=0.12,
                noise_min=0.02,
                decay_power=2.0,
                service_rate_ema=0.0,
                service_target=0.85,
                service_boost=0.05,
            )
            self.assertGreaterEqual(value.final_sigma, 0.02)
            self.assertLessEqual(value.final_sigma, 0.17)


if __name__ == "__main__":
    unittest.main()
