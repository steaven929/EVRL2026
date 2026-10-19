import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from specific.dclily_elec import dclily


class MultiDayDataSplitTests(unittest.TestCase):
    def test_sip_config_uses_twenty_five_train_and_five_complete_test_days(self):
        config_path = Path(__file__).parent / "data" / "SIP" / "config_sip.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["TRAIN_DAY_INDICES"], list(range(1, 26)))
        self.assertEqual(config["TEST_DAY_INDICES"], list(range(26, 31)))
        # V4.7 historical main-method baseline: 144 chargers / 250 spaces.
        self.assertEqual(sum(len(zone["charging_slots"]) for zone in config["ZONES"]), 144)

    def test_all_thirty_days_have_positive_demand_in_every_timestep(self):
        data_root = Path(__file__).parent / "data" / "SIP"
        totals = []
        for day in range(1, 31):
            frame = pd.read_csv(data_root / f"od_demand_{day}.csv")
            demand = frame.iloc[:, 2:].to_numpy(dtype=float)
            totals.append(float(demand.sum()))
            self.assertGreater(demand.sum(), 0.0, f"day {day} has zero demand")
            self.assertTrue(
                np.all(demand.sum(axis=0) > 0),
                f"day {day} contains a zero-demand timestep",
            )
        self.assertGreater(np.std(totals), 0.0, "all daily demand totals are identical")

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
