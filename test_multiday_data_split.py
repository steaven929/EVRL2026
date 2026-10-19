import json
import unittest
from pathlib import Path

import numpy as np

from specific.dclily_elec import dclily


class MultiDayDataSplitTests(unittest.TestCase):
    def test_sip_config_uses_twenty_five_train_and_five_complete_test_days(self):
        config_path = Path(__file__).parent / "data" / "SIP" / "config_sip.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["TRAIN_DAY_INDICES"], list(range(1, 26)))
        self.assertEqual(config["TEST_DAY_INDICES"], list(range(26, 31)))
        self.assertEqual(sum(len(zone["charging_slots"]) for zone in config["ZONES"]), 75)

    def test_environment_loads_selected_distinct_test_day(self):
        env = dclily()
        env.set_training_mode(0)
        env.set_test_day(26)
        env.reset()
        day_26 = env.od_demand_left.copy()
        self.assertEqual(env.current_demand_day, 26)

        env.set_test_day(27)
        env.reset()
        day_27 = env.od_demand_left.copy()
        self.assertEqual(env.current_demand_day, 27)
        self.assertFalse(np.array_equal(day_26, day_27))


if __name__ == "__main__":
    unittest.main()
