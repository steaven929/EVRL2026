import tempfile
import unittest
from pathlib import Path

import pandas as pd

from specific.sip_demand_conversion import convert_demand_directory


class SipDemandConversionTests(unittest.TestCase):
    def test_converts_each_raw_day_to_one_complete_wide_day(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            output_dir = root / "output"
            raw_dir.mkdir()

            rows = []
            for origin in range(2):
                for destination in range(2):
                    for timestep in range(3):
                        demand = 0 if origin == destination else 100 * origin + 10 * destination + timestep
                        rows.append([0, origin, destination, timestep, demand])
            pd.DataFrame(rows).to_csv(raw_dir / "q_0.csv", header=False, index=False)

            outputs = convert_demand_directory(
                raw_dir,
                output_dir,
                day_count=1,
                zone_count=2,
                timestep_count=3,
            )

            self.assertEqual(outputs, [output_dir / "od_demand_1.csv"])
            converted = pd.read_csv(outputs[0])
            self.assertEqual(list(converted.columns), ["O", "D", "0", "1", "2"])
            self.assertEqual(converted[["O", "D"]].values.tolist(), [[0, 1], [1, 0]])
            self.assertEqual(converted[["0", "1", "2"]].values.tolist(), [[10, 11, 12], [100, 101, 102]])

    def test_rejects_a_source_that_is_only_one_nonzero_timestep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            rows = []
            for origin in range(2):
                for destination in range(2):
                    for timestep in range(3):
                        demand = 1 if origin != destination and timestep == 0 else 0
                        rows.append([0, origin, destination, timestep, demand])
            pd.DataFrame(rows).to_csv(raw_dir / "q_0.csv", header=False, index=False)
            with self.assertRaisesRegex(ValueError, "not a complete demand day"):
                convert_demand_directory(
                    raw_dir, root / "output", day_count=1,
                    zone_count=2, timestep_count=3,
                )


if __name__ == "__main__":
    unittest.main()
