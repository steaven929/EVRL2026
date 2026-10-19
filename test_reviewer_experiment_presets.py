import unittest
import io
import sys
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np

from core.marl_training import marl_test_days
from core.reward_components import build_reward_components
from core.util import get_args, resolve_experiment_preset
from main_marl import build_output_directory_name
from specific.dclily_elec import dclily


class ReviewerPresetTests(unittest.TestCase):
    def test_output_directory_keeps_run_metadata_and_adds_experiment_identity(self):
        name = build_output_directory_name(
            {
                "experiment_id": "E2",
                "experiment_variant": "full",
                "seed": 1,
                "fleet_size": 200,
                "train_eps": 500,
            },
            "20261017-140500",
        )
        self.assertEqual(
            name,
            "E2_full_seed1_200car_marl_500ep_20261017-140500",
        )

    def test_e2_ablation_presets_change_exactly_one_switch(self):
        full = resolve_experiment_preset("E2", "full")
        cases = {
            "no_zone_embedding": "use_zone_embedding",
            "no_global_context": "use_global_context",
            "no_parameter_sharing": "share_policy",
        }
        for variant, changed_key in cases.items():
            preset = resolve_experiment_preset("E2", variant)
            for key in ("use_zone_embedding", "use_global_context", "share_policy"):
                expected = False if key == changed_key else full[key]
                self.assertEqual(preset[key], expected)
        linear = resolve_experiment_preset("E2", "no_adaptive_exploration")
        self.assertEqual(linear["exploration_schedule"], "linear")

    def test_e3_exploration_presets_cover_requested_strategies(self):
        for variant in ("adaptive", "fixed", "linear"):
            preset = resolve_experiment_preset("E3", variant)
            self.assertEqual(preset["exploration_schedule"], variant)

    def test_e4_uses_twenty_five_train_and_five_disjoint_test_days(self):
        preset = resolve_experiment_preset("E4", "five_days")
        self.assertEqual(preset["train_day_indices"], list(range(1, 26)))
        self.assertEqual(preset["test_day_indices"], list(range(26, 31)))
        self.assertTrue(set(preset["train_day_indices"]).isdisjoint(preset["test_day_indices"]))

    def test_e5_fleet_presets_cover_requested_sizes(self):
        for size in (200, 1000, 2000):
            preset = resolve_experiment_preset("E5", f"fleet_{size}")
            self.assertEqual(preset["fleet_size"], size)

    def test_e7_charging_presets_cover_p1_legacy_and_none(self):
        expected = {"optimized": "p1", "original_rule": "legacy", "no_charging": "none"}
        for variant, mode in expected.items():
            self.assertEqual(resolve_experiment_preset("E7", variant)["charging_assignment_mode"], mode)

    def test_zhu_preset_fails_instead_of_relabelling_hmarl(self):
        with self.assertRaisesRegex(NotImplementedError, "Zhu"):
            resolve_experiment_preset("E1", "zhu2025")

    def test_cli_selector_expands_e5_preset_before_full_parse(self):
        argv = ["program", "--experiment_id", "E5", "--experiment_variant", "fleet_1000"]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            args = get_args()
        self.assertEqual(args["fleet_size"], 1000)
        self.assertEqual(args["experiment_name"], "车队规模扩展（200/1000/2000辆）")
        self.assertEqual(args["experiment_variant_name"], "1000辆车规模")
        self.assertEqual(args["train_day_indices"], list(range(1, 26)))
        self.assertEqual(args["test_day_indices"], list(range(26, 31)))

    def test_training_defaults_match_results_v1_main_marl_reference(self):
        with patch.object(sys, "argv", ["program"]), redirect_stdout(io.StringIO()):
            args = get_args()
        expected = {
            "test_eps": 10,
            "fleet_size": 200,
            "n_agents": 50,
            "actor_lr": 5e-5,
            "critic_lr": 1e-4,
            "gamma": 0.95,
            "tau": 5e-4,
            "batch_size": 128,
            "memory_capacity": 64000,
            "actor_hidden_dim": 256,
            "critic_hidden_dim": 256,
            "reward_scale": 10000.0,
        }
        for key, value in expected.items():
            self.assertEqual(args[key], value, key)
        self.assertGreater(args["train_eps"], 0)


class RuntimeExperimentSwitchTests(unittest.TestCase):
    def test_runtime_day_split_accepts_twenty_five_train_and_five_test_days(self):
        env = dclily()
        env.set_day_split(range(1, 26), range(26, 31))
        env.set_training_mode(0)
        env.set_test_day(26)
        env.reset()
        self.assertEqual(env.current_demand_day, 26)

    def test_runtime_fleet_scaling_updates_cars_parking_and_chargers(self):
        env = dclily()
        self.addCleanup(env.set_fleet_size, 200)
        env.set_fleet_size(1000)
        allocation = env.set_charger_ratio(0.30)
        env.set_training_mode(0)
        env.set_test_day(26)
        env.reset()
        self.assertEqual(env.CAR_NUM, 1000)
        self.assertEqual(len(env.car), 1000)
        self.assertGreaterEqual(int(np.max(env.observation_space.nvec)), 1001)
        self.assertEqual(int(np.sum(env.available[0]) + np.sum(env.available[1])), 1250)
        self.assertEqual(sum(allocation), 375)

    def test_no_charging_mode_returns_no_assignments(self):
        env = dclily.__new__(dclily)
        env.set_charging_assignment_mode("none")
        result = env.check_idle_vehicles(np.zeros(50), {})
        self.assertEqual(result, {})
        self.assertEqual(env.last_p1_stats["status_name"], "NO_CHARGING")


class _Agent:
    def sample_action(self, obs_n, global_state=None, add_noise=False):
        return np.zeros((2, 3), dtype=np.float32)


class _TimedEnv:
    n_agents = 2
    TIME_NUM = 1
    CAR_NUM = 200

    def set_training_mode(self, mode):
        pass

    def set_test_day(self, day):
        self.day = day

    def reset(self):
        self.step_index = 0
        return np.zeros(11, dtype=np.float32)

    def get_agent_obs(self, agent_id):
        return np.array([0, 0, 0, 0, 0, 0, agent_id], dtype=np.float32)

    def step(self, actions):
        self.step_index += 1
        info = build_reward_components(
            service_revenue=1, relocation_cost=0, p_soc=0, p_svr=0,
            b_chg=0, b_soc=0, environment_reward=1, reward_scale=1,
        )
        info.update({"served_orders": 1, "total_demand": 1, "solver_stats": []})
        return np.zeros(11, dtype=np.float32), 1.0, True, info


class RuntimeLoggingTests(unittest.TestCase):
    def test_multiday_output_contains_actor_and_total_decision_time(self):
        writer = Mock()
        with patch("core.marl_training.EvaluationWriter", return_value=writer):
            marl_test_days(
                _TimedEnv(), _Agent(),
                {"test_day_indices": [26], "method": "efficiency", "seed": 1,
                 "charger_ratio": 0.3, "reward_scale": 1.0},
                output_root="unused", experiment="E6",
            )
            row = writer.write_day.call_args.kwargs["timestep_rows"][0]
            self.assertIn("actor_inference_time_seconds", row)
            self.assertIn("environment_step_time_seconds", row)
            self.assertIn("total_decision_time_seconds", row)


if __name__ == "__main__":
    unittest.main()
