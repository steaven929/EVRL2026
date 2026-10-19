import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.reward_components import (
    REWARD_COMPONENT_KEYS,
    build_reward_components,
    format_reward_breakdown,
    write_reward_components_csv,
    write_reward_formula_summary,
)
from core.marl_training import marl_train


class _Memory:
    def push(self, **kwargs):
        pass

    def __len__(self):
        return 0


class _Agent:
    def __init__(self):
        self.memory = _Memory()

    def sample_action(self, obs_n, global_state, add_noise, noise_scale):
        return np.zeros((2, 3), dtype=np.float32)


class _Env:
    n_agents = 2
    TIME_NUM = 2

    def __init__(self):
        self.step_index = 0

    def set_experiment_variant(self, value):
        pass

    def set_reward_mode(self, value):
        pass

    def set_epoch(self, value):
        pass

    def reset(self):
        self.step_index = 0
        return np.zeros(3, dtype=np.float32)

    def get_agent_obs(self, agent_id):
        return np.zeros(7, dtype=np.float32)

    def step(self, actions):
        self.step_index += 1
        reward = 40.0 * self.step_index
        info = build_reward_components(
            service_revenue=100.0 * self.step_index,
            relocation_cost=10.0 * self.step_index,
            p_soc=5.0,
            p_svr=2.0,
            b_chg=3.0,
            b_soc=4.0,
            environment_reward=reward,
            reward_scale=1.0,
        )
        return np.zeros(3, dtype=np.float32), reward, self.step_index == 2, info


class RewardComponentTrackingTests(unittest.TestCase):
    def test_reward_breakdown_prints_nor_pft_and_scaled_reward_formulas(self):
        lines = format_reward_breakdown(
            {
                "service_revenue": 120.0,
                "relocation_cost": 20.0,
                "NOR": 100.0,
                "P_soc": 7.0,
                "P_svr": 3.0,
                "B_chg": 15.0,
                "B_soc": 5.0,
                "PFT": 110.0,
                "environment_reward": 42.0,
                "scaled_training_reward": 4.2,
            },
            reward_scale=10.0,
        )
        self.assertIn("NOR = service_revenue(120.00) - relocation_cost(20.00) = 100.00", lines[0])
        self.assertIn(
            "PFT = NOR(100.00) - P_soc(7.00) - P_svr(3.00) + B_chg(15.00) + B_soc(5.00) = 110.00",
            lines[1],
        )
        self.assertIn(
            "scaled_training_reward = environment_reward(42.00) / reward_scale(10.00) = 4.200000",
            lines[2],
        )

    def test_build_components_keeps_paper_pft_separate_from_training_reward(self):
        values = build_reward_components(
            service_revenue=120.0,
            relocation_cost=20.0,
            p_soc=7.0,
            p_svr=3.0,
            b_chg=15.0,
            b_soc=5.0,
            environment_reward=42.0,
            reward_scale=10.0,
        )

        self.assertEqual(tuple(values), REWARD_COMPONENT_KEYS)
        self.assertEqual(values["NOR"], 100.0)
        self.assertEqual(values["PFT"], 110.0)
        self.assertEqual(values["environment_reward"], 42.0)
        self.assertEqual(values["scaled_training_reward"], 4.2)

    def test_write_csv_writes_episode_and_timestep_rows(self):
        arrays = {
            key: np.array([[index + 1.0, index + 2.0], [index + 3.0, np.nan]])
            for index, key in enumerate(REWARD_COMPONENT_KEYS)
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "reward_components_by_timestep.csv"
            write_reward_components_csv(output_path, arrays)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(list(rows[0]), ["episode", "timestep", *REWARD_COMPONENT_KEYS])
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["episode"], "1")
        self.assertEqual(rows[0]["timestep"], "1")
        self.assertEqual(rows[-1]["episode"], "2")
        self.assertEqual(rows[-1]["timestep"], "2")
        self.assertEqual(rows[-1]["service_revenue"].lower(), "nan")

    def test_write_reward_formula_summary_saves_episode_level_subterms(self):
        arrays = {
            key: np.array([[index + 1.0, index + 2.0], [index + 3.0, index + 4.0]])
            for index, key in enumerate(REWARD_COMPONENT_KEYS)
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "reward_formula_breakdown_by_episode.csv"
            write_reward_formula_summary(output_path, arrays, reward_scale=10.0)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode"], "1")
        self.assertIn("service_revenue", rows[0])
        self.assertIn("PFT", rows[0])
        self.assertIn("environment_reward", rows[0])
        self.assertIn("reward_minus_PFT", rows[0])
        self.assertIn("pft_identity_error", rows[0])
        self.assertIn("reward_scaling_error", rows[0])

    @patch("core.marl_training.compute_all_improved_metrics", return_value={})
    @patch("core.marl_training.compute_all_metrics", return_value={})
    def test_marl_train_records_timestep_arrays_and_episode_totals(self, *_):
        result = marl_train(
            _Env(),
            _Agent(),
            {
                "train_eps": 1,
                "TIME_NUM": 2,
                "reward_scale": 10.0,
                "batch_size": 128,
                "exp_variant": "a1",
                "reward_mode": "shaping",
            },
        )

        for key in REWARD_COMPONENT_KEYS:
            self.assertEqual(result[f"{key}_by_timestep"].shape, (1, 2), key)
            self.assertEqual(result[f"{key}_history"].shape, (1,), key)
        np.testing.assert_allclose(result["environment_reward_by_timestep"], [[40.0, 80.0]])
        np.testing.assert_allclose(result["scaled_training_reward_by_timestep"], [[4.0, 8.0]])
        self.assertEqual(result["environment_reward_history"][0], 120.0)
        self.assertEqual(result["scaled_training_reward_history"][0], 12.0)


if __name__ == "__main__":
    unittest.main()
