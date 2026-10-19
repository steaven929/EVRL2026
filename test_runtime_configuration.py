import io
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from core.console import configure_utf8_console
from core.output_paths import RESULTS_ROOT, result_run_directory
from core.util import get_args, str_to_bool
from specific.dclily_elec import CHARGER_PER_ZONE, dclily


class RuntimeConfigurationTest(unittest.TestCase):
    def test_all_training_runs_use_resluts2610_root(self):
        expected_root = Path(__file__).resolve().parent / "resluts2610"
        self.assertEqual(RESULTS_ROOT, expected_root)
        self.assertEqual(
            result_run_directory("E2_full_seed1_200car_marl_1200ep_timestamp"),
            expected_root / "E2_full_seed1_200car_marl_1200ep_timestamp",
        )

    def tearDown(self):
        env = dclily.__new__(dclily)
        env.config = {}
        env.set_charger_ratio(0.30)

    def test_console_configuration_enables_immediate_line_output(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        configure_utf8_console(stream)
        self.assertEqual(stream.encoding.lower().replace("-", ""), "utf8")
        self.assertTrue(stream.line_buffering)
        self.assertTrue(stream.write_through)

    def test_get_args_prints_list_defaults_without_format_error(self):
        with patch.object(sys, "argv", ["program"]), redirect_stdout(io.StringIO()):
            args = get_args()
        self.assertEqual(args["test_day_indices"], [26, 27, 28, 29, 30])

    def test_boolean_parser_handles_false_explicitly(self):
        self.assertFalse(str_to_bool("false"))
        self.assertFalse(str_to_bool("0"))
        self.assertTrue(str_to_bool("true"))

    def test_sensitivity_ratios_rebuild_exact_charger_totals(self):
        env = dclily.__new__(dclily)
        env.config = {}
        expected = {0.300: 75, 0.432: 108, 0.576: 144}
        for ratio, total in expected.items():
            allocation = env.set_charger_ratio(ratio)
            self.assertEqual(sum(allocation), total)
            self.assertEqual(sum(map(len, CHARGER_PER_ZONE)), total)
            self.assertTrue(all(count <= 5 for count in allocation))

    def test_144_charger_baseline_covers_every_zone_nearly_uniformly(self):
        env = dclily.__new__(dclily)
        env.config = {}
        allocation = env.set_charger_ratio(0.576)
        self.assertEqual(allocation.count(3), 44)
        self.assertEqual(allocation.count(2), 6)
        self.assertEqual(min(allocation), 2)
        self.assertEqual(sum(allocation), 144)

    def test_assignment_mode_rejects_unknown_value(self):
        env = dclily.__new__(dclily)
        env.set_charging_assignment_mode("legacy")
        self.assertEqual(env.charging_assignment_mode, "legacy")
        with self.assertRaises(ValueError):
            env.set_charging_assignment_mode("unknown")


if __name__ == "__main__":
    unittest.main()
