import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.evaluation_logging import write_exploration_history
from core import marl_training
from core.reward_components import build_reward_components


class _Agent:
    def sample_action(self, obs_n, global_state=None, add_noise=False):
        return np.zeros((2, 3), dtype=np.float32)


class _Env:
    n_agents = 2
    TIME_NUM = 2
    CAR_NUM = 200

    def __init__(self):
        self.day = 26
        self.step_index = 0

    def set_training_mode(self, mode):
        self.mode = mode

    def set_test_day(self, day):
        self.day = day

    def set_experiment_variant(self, value):
        pass

    def set_reward_mode(self, value):
        pass

    def set_charging_assignment_mode(self, value):
        pass

    def set_charger_ratio(self, value):
        pass

    def reset(self):
        self.step_index = 0
        return np.zeros(11, dtype=np.float32)

    def get_agent_obs(self, agent_id):
        return np.zeros(7, dtype=np.float32)

    def step(self, actions):
        self.step_index += 1
        reward = 100.0 * self.step_index
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
        # The environment currently emits NaN here; evaluation must replace it
        # with the configured training-scale value before persistence.
        info["scaled_training_reward"] = np.nan
        info.update({
            "served_orders": 8.0,
            "total_demand": 10.0,
            "operating_profit": 85.0,
            "svc": 0.8,
            "rld": 12.0 * self.step_index,
            "srd": 6.0 * self.step_index,
            "demand_matrix": np.ones((2, 2)),
            "demand_orders": np.eye(2),
            "charging_actions": {0: 1},
            "parking_available_by_zone": np.array([2.0, 3.0]),
            "parking_used_by_zone": np.array([3.0, 2.0]),
            "charger_available_by_zone": np.array([1.0, 1.0]),
            "charger_used_by_zone": np.array([1.0, 0.0]),
            "charging_events": [],
            "solver_stats": [],
            "charging_grid_energy_kwh": 2.0 * self.step_index,
            "charging_battery_energy_kwh": 1.8 * self.step_index,
            "charging_energy_cost": 3.0 * self.step_index,
            "charging_travel_distance_km": 4.0 * self.step_index,
            "charging_travel_cost": 5.0 * self.step_index,
            "charging_completion_count": self.step_index,
        })
        return np.zeros(11, dtype=np.float32), reward, self.step_index == 2, info


class ReviewerResultPersistenceTests(unittest.TestCase):
    def test_exploration_histories_are_saved_as_arrays_and_csv(self):
        values = {
            "exploration_base_sigma_history": [0.12, 0.10],
            "exploration_service_ema_history": [0.60, 0.70],
            "exploration_service_gap_history": [0.25, 0.15],
            "exploration_boost_history": [0.0125, 0.0075],
            "exploration_sigma_history": [0.1325, 0.1075],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            write_exploration_history(temp_dir, values)
            for key, expected in values.items():
                np.testing.assert_allclose(
                    np.load(Path(temp_dir) / f"{key}.npy"), expected
                )
            with (Path(temp_dir) / "exploration_history.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode"], "1")
        self.assertEqual(float(rows[1]["final_sigma"]), 0.1075)

    @patch("core.marl_training.compute_all_metrics")
    def test_test_day_saves_scaled_reward_and_charging_metrics(self, metrics):
        metrics.side_effect = [
            {
                "charging_efficiency": 0.4,
                "vehicle_utilization": 0.5,
                "fleet_health": 0.6,
                "zone_coordination": 0.7,
                "response_readiness": 0.8,
                "charger_occupancy_rate": 0.2,
                "charger_queue_delay": 4.0,
                "energy_not_served": 2.0,
                "safety_soc_violation_count": 3.0,
                "charge_shift_success_rate": 0.9,
            },
            {
                "charging_efficiency": 0.6,
                "vehicle_utilization": 0.7,
                "fleet_health": 0.8,
                "zone_coordination": 0.9,
                "response_readiness": 1.0,
                "charger_occupancy_rate": 0.4,
                "charger_queue_delay": 6.0,
                "energy_not_served": 4.0,
                "safety_soc_violation_count": 5.0,
                "charge_shift_success_rate": 0.7,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            marl_training.marl_test_days(
                _Env(), _Agent(),
                {
                    "test_day_indices": [26],
                    "method": "hmarl-rc",
                    "seed": 1,
                    "charger_ratio": 0.30,
                    "reward_scale": 10.0,
                    "charging_assignment_mode": "p1",
                },
                output_root=Path(temp_dir), experiment="review",
            )
            day_dir = Path(temp_dir) / "review" / "hmarl-rc" / "200" / "1" / "ratio_0.30" / "day_26"
            with (day_dir / "reward_components.csv").open(encoding="utf-8", newline="") as handle:
                reward_rows = list(csv.DictReader(handle))
            with (day_dir / "timestep_metrics.csv").open(encoding="utf-8", newline="") as handle:
                timestep_rows = list(csv.DictReader(handle))
            summary = json.loads((day_dir / "summary.json").read_text("utf-8"))
            with (day_dir / "vehicle_charging_events.csv").open(encoding="utf-8", newline="") as handle:
                empty_event_reader = csv.DictReader(handle)
                empty_event_rows = list(empty_event_reader)

        self.assertEqual([float(row["scaled_training_reward"]) for row in reward_rows], [10.0, 20.0])
        self.assertEqual([float(row["charging_efficiency"]) for row in timestep_rows], [0.4, 0.6])
        self.assertAlmostEqual(summary["charging_efficiency"], 0.5)
        self.assertAlmostEqual(summary["energy_not_served"], 3.0)
        self.assertAlmostEqual(summary["energy_not_served_total"], 6.0)
        self.assertAlmostEqual(summary["service_revenue"], 200.0)
        self.assertAlmostEqual(summary["relocation_cost"], 20.0)
        self.assertAlmostEqual(summary["NOR"], 180.0)
        self.assertAlmostEqual(summary["PFT"], 180.0)
        self.assertAlmostEqual(summary["environment_reward"], 300.0)
        self.assertAlmostEqual(summary["scaled_training_reward"], 30.0)
        self.assertAlmostEqual(summary["reward_minus_PFT"], 120.0)
        self.assertAlmostEqual(summary["pft_identity_error"], 0.0)
        self.assertAlmostEqual(summary["reward_scaling_error"], 0.0)
        self.assertAlmostEqual(summary["svc"], 0.8)
        self.assertAlmostEqual(summary["rld"], 18.0)
        self.assertAlmostEqual(summary["srd"], 9.0)
        self.assertEqual(empty_event_rows, [])
        self.assertIn("event", empty_event_reader.fieldnames)
        self.assertIn("charging_cost_rmb", empty_event_reader.fieldnames)

    def test_training_charging_summary_aggregates_each_episode(self):
        self.assertTrue(hasattr(marl_training, "summarize_training_charging"))
        summary = marl_training.summarize_training_charging(
            episode=3,
            demand_days=[7, 7],
            step_metrics=[
                {"charging_grid_energy_kwh": 2.0, "charging_travel_distance_km": 1.5,
                 "charging_travel_cost": 4.0, "charging_completion_count": 1},
                {"charging_grid_energy_kwh": 3.0, "charging_travel_distance_km": 2.5,
                 "charging_travel_cost": 6.0, "charging_completion_count": 2},
            ],
            events=[{"event": "reserved"}, {"event": "charging"}, {"event": "charging"}],
        )
        self.assertEqual(summary["episode"], 3)
        self.assertEqual(summary["demand_day"], 7)
        self.assertAlmostEqual(summary["charging_grid_energy_kwh"], 5.0)
        self.assertAlmostEqual(summary["charging_travel_distance_km"], 4.0)
        self.assertAlmostEqual(summary["charging_travel_cost"], 10.0)
        self.assertEqual(summary["charging_completion_count"], 3)
        self.assertEqual(summary["charging_reservation_count"], 1)
        self.assertEqual(summary["charging_event_count"], 3)


if __name__ == "__main__":
    unittest.main()
