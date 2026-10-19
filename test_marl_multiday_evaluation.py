import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.marl_training import marl_test_days
from core.reward_components import build_reward_components


class _Agent:
    def sample_action(self, obs_n, global_state=None, add_noise=False):
        return np.zeros((2, 3), dtype=np.float32)


class _Env:
    n_agents = 2
    TIME_NUM = 2
    CAR_NUM = 200

    def __init__(self):
        self.day = None
        self.step_index = 0

    def set_training_mode(self, mode):
        self.mode = mode

    def set_test_day(self, day):
        self.day = day

    def set_experiment_variant(self, value):
        pass

    def set_reward_mode(self, value):
        pass

    def reset(self):
        self.step_index = 0
        return np.zeros(11, dtype=np.float32)

    def get_agent_obs(self, agent_id):
        return np.array([0, 0, 0, 0, 0, 0, agent_id], dtype=np.float32)

    def step(self, actions):
        self.step_index += 1
        reward = float(self.day + self.step_index)
        info = build_reward_components(
            service_revenue=100.0,
            relocation_cost=10.0,
            p_soc=5.0,
            p_svr=2.0,
            b_chg=3.0,
            b_soc=4.0,
            environment_reward=reward,
            reward_scale=1.0,
        )
        info.update({
            "demand_day": self.day,
            "service_rate": 0.8,
            "served_orders": 8.0,
            "total_demand": 10.0,
            "operating_profit": 85.0,
            "demand_matrix": np.ones((2, 2)),
            "demand_orders": np.eye(2),
            "charging_actions": {0: 1},
            "parking_available_by_zone": np.array([2.0, 3.0]),
            "parking_used_by_zone": np.array([3.0, 2.0]),
            "charger_available_by_zone": np.array([1.0, 1.0]),
            "charger_used_by_zone": np.array([1.0, 0.0]),
            "charging_events": [{"vehicle_id": 1, "event": "reserved"}],
            "solver_stats": [{"optimizer": "P1", "status_name": "FALLBACK"}],
        })
        done = self.step_index == self.TIME_NUM
        return np.zeros(11, dtype=np.float32), reward, done, info


class MultiDayEvaluationTests(unittest.TestCase):
    @patch("core.marl_training.compute_all_metrics", return_value={})
    def test_runner_saves_all_five_distinct_days(self, _):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = marl_test_days(
                _Env(),
                _Agent(),
                {
                    "test_day_indices": [26, 27, 28, 29, 30],
                    "method": "hmarl-rc",
                    "seed": 11,
                    "charger_ratio": 0.30,
                    "reward_scale": 1.0,
                    "exp_variant": "a1",
                    "reward_mode": "base",
                },
                output_root=Path(temp_dir),
                experiment="review",
            )
            self.assertEqual(sorted(result), [26, 27, 28, 29, 30])
            for day in range(26, 31):
                day_dir = (
                    Path(temp_dir) / "review" / "hmarl-rc" / "200" / "11"
                    / "ratio_0.30" / f"day_{day}"
                )
                self.assertTrue((day_dir / "timestep_metrics.csv").exists())
                self.assertTrue((day_dir / "solver_stats.csv").exists())


if __name__ == "__main__":
    unittest.main()
