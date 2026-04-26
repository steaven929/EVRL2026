import unittest
from unittest import mock

import numpy as np

from core import marl_training


class _DummyMemory:
    def __init__(self):
        self._items = []

    def push(self, **kwargs):
        self._items.append(kwargs)

    def __len__(self):
        return len(self._items)


class _DummyAgent:
    def __init__(self):
        self.memory = _DummyMemory()

    def sample_action(self, obs_n, global_state=None, add_noise=False, noise_scale=0.0):
        return np.zeros((obs_n.shape[0], 3), dtype=np.float32)

    def update(self):
        return 1.0, 2.0

    def save_model(self, path):
        return None


class _DummyEnv:
    def __init__(self):
        self.n_agents = 2
        self.ZONE_NUM = 2
        self.TIME_NUM = 2
        self.CAR_NUM = 2
        self.BATTERY_CAPACITY = 100.0
        self.TIME_LENGTH = 10.0
        self.car = [
            [0, 80.0, 0, -1],
            [1, 40.0, 1, -1],
        ]
        self.charging_stations = {
            0: [{"occupied_by": None}],
            1: [{"occupied_by": None}],
        }
        self.current_step_logs = {
            0: {"decision_charge": True, "shifted": True},
            1: {"decision_charge": False, "shifted": False},
        }
        self._step_in_episode = 0

    def set_training_mode(self, mode):
        return None

    def set_epoch(self, epoch):
        return None

    def set_experiment_variant(self, variant):
        return None

    def set_reward_mode(self, reward_mode):
        return None

    def reset(self):
        self._step_in_episode = 0
        self.car = [
            [0, 80.0, 0, -1],
            [1, 40.0, 1, -1],
        ]
        return np.zeros(11, dtype=np.float32)

    def get_agent_obs(self, agent_id):
        return np.array([0, 0, 0, 0, 0, 0, float(agent_id)], dtype=np.float32)

    def step(self, actions):
        self._step_in_episode += 1
        done = self._step_in_episode >= self.TIME_NUM
        info = {
            "svc": 1.0,
            "rld": 2.0,
            "srd": 3.0,
            "service_rate": 0.5,
            "total_demand": 4.0,
            "served_orders": 3.0,
            "demand_matrix": np.array([[0.0, 2.0], [1.0, 1.0]], dtype=float),
            "demand_orders": np.array([[0.0, 1.0], [1.0, 1.0]], dtype=float),
            "charging_actions": {0: 1.0, 1: 0.0},
            "action_table_true": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float),
            "parking_available_by_zone": np.array([2.0, 2.0], dtype=float),
            "parking_used_by_zone": np.array([1.0, 0.0], dtype=float),
            "charger_available_by_zone": np.array([1.0, 1.0], dtype=float),
            "charger_used_by_zone": np.array([1.0, 0.0], dtype=float),
            "dispatch_orders": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float),
        }
        self.car = [
            [0, 75.0, 0, 0],
            [1, 35.0, 0, -1],
        ]
        return np.zeros(11, dtype=np.float32), 10.0, done, info


def _base_cfg():
    return {
        "train_eps": 2,
        "test_eps": 1,
        "max_steps": 2,
        "reward_scale": 1.0,
        "exp_variant": "a1",
        "reward_mode": "base",
        "marl_noise_start": 0.3,
        "marl_noise_min": 0.1,
        "marl_noise_decay_power": 1.0,
        "marl_noise_warmup_eps": 0,
        "marl_service_target": 0.85,
        "marl_noise_service_boost": 0.0,
        "marl_learning_starts": 1,
        "marl_updates_per_step": 1,
        "eval_every": 1,
        "eval_eps": 1,
        "batch_size": 1,
    }


class MarlOutputExportTests(unittest.TestCase):
    @mock.patch.object(
        marl_training,
        "compute_all_metrics",
        lambda env, info: {
            "charging_efficiency": 0.75,
            "vehicle_utilization": 0.25,
            "fleet_health": 0.6,
            "zone_coordination": 0.5,
            "response_readiness": 0.4,
            "charger_occupancy_rate": 0.5,
            "charger_queue_delay": 0.0,
            "energy_not_served": 1.0,
            "safety_soc_violation_count": 0.0,
            "charge_shift_success_rate": 1.0,
        },
    )
    @mock.patch.object(
        marl_training,
        "compute_all_improved_metrics",
        lambda env, info: {
            "weighted_service_rate": 0.5,
            "effective_service_rate": 0.6,
            "service_quality_index": 0.7,
            "dispatch_efficiency": 0.8,
            "effective_dispatch_rate": 0.9,
            "dispatch_roi": 1.0,
        },
    )
    def test_marl_train_returns_full_metric_export_set(self):
        env = _DummyEnv()
        agent = _DummyAgent()
        cfg = _base_cfg()

        res = marl_training.marl_train(env, agent, cfg)

        expected_keys = {
            "rewards",
            "raw_rewards",
            "ma_rewards",
            "critic_losses",
            "actor_losses",
            "rld_history",
            "svc_history",
            "srd_history",
            "charging_efficiency_history",
            "energy_not_served_history",
            "charger_queue_delay_history",
            "safety_soc_violation_count_history",
            "vehicle_utilization_history",
            "fleet_health_history",
            "zone_coordination_history",
            "response_readiness_history",
            "charger_occupancy_rate_history",
            "charge_shift_success_rate_history",
            "weighted_service_rate_history",
            "effective_service_rate_history",
            "service_quality_index_history",
            "dispatch_efficiency_history",
            "effective_dispatch_rate_history",
            "dispatch_roi_history",
            "svc_by_timestep",
            "rld_by_timestep",
            "srd_by_timestep",
            "reward_by_timestep",
            "total_demand_by_timestep",
            "served_orders_by_timestep",
            "charging_efficiency_by_timestep",
            "charger_queue_delay_by_timestep",
            "energy_not_served_by_timestep",
            "soc_violation_by_timestep",
            "svc_timestep_mean",
            "rld_timestep_mean",
            "srd_timestep_mean",
            "reward_timestep_mean",
            "total_demand_timestep_mean",
            "served_orders_timestep_mean",
            "charging_efficiency_timestep_mean",
            "charger_queue_delay_timestep_mean",
            "energy_not_served_timestep_mean",
            "soc_violation_timestep_mean",
            "zone_ids",
            "timestep_ids",
            "demand_by_timestep_zone",
            "served_by_timestep_zone",
            "service_rate_by_timestep_zone",
            "charging_actions_by_timestep_zone",
            "dispatch_by_timestep_zone",
            "demand_timestep_zone_mean",
            "served_timestep_zone_mean",
            "service_rate_timestep_zone_mean",
            "charging_actions_timestep_zone_mean",
            "dispatch_timestep_zone_mean",
            "parking_available_by_timestep_zone",
            "parking_used_by_timestep_zone",
            "charger_available_by_timestep_zone",
            "charger_used_by_timestep_zone",
            "parking_available_timestep_zone_mean",
            "parking_used_timestep_zone_mean",
            "charger_available_timestep_zone_mean",
            "charger_used_timestep_zone_mean",
            "eval_rewards",
            "eval_episodes",
        }
        self.assertTrue(expected_keys.issubset(set(res.keys())))
        self.assertEqual(res["svc_by_timestep"].shape, (cfg["train_eps"], env.TIME_NUM))
        self.assertEqual(res["demand_by_timestep_zone"].shape, (cfg["train_eps"], env.TIME_NUM, env.ZONE_NUM))
        self.assertEqual(list(res["zone_ids"]), [0, 1])
        self.assertEqual(list(res["timestep_ids"]), [0, 1])


if __name__ == "__main__":
    unittest.main()
