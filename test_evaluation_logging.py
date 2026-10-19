import csv
import json
import tempfile
import unittest
from pathlib import Path

from core.evaluation_logging import EvaluationIdentity, EvaluationWriter


class EvaluationLoggingTests(unittest.TestCase):
    def test_each_test_day_has_independent_raw_files_and_manifest_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = EvaluationWriter(Path(temp_dir), experiment="p1-review")
            for day in range(26, 31):
                identity = EvaluationIdentity(
                    method="hmarl-rc",
                    fleet_size=200,
                    seed=11,
                    charger_ratio=0.30,
                    test_day=day,
                )
                day_dir = writer.write_day(
                    identity=identity,
                    config={"test_day": day},
                    summary={"environment_reward": float(day)},
                    timestep_rows=[{"timestep": 0, "reward": float(day)}],
                    zone_timestep_rows=[{"timestep": 0, "zone": 0, "served": day}],
                    charging_event_rows=[{"timestep": 0, "vehicle_id": 1, "event": "reserved"}],
                    reward_rows=[{"timestep": 0, "PFT": float(day)}],
                    solver_rows=[{"timestep": 0, "optimizer": "P1", "status": "fallback"}],
                )
                self.assertEqual(day_dir.name, f"day_{day}")
                for filename in (
                    "config.json",
                    "summary.json",
                    "timestep_metrics.csv",
                    "zone_timestep_metrics.csv",
                    "vehicle_charging_events.csv",
                    "reward_components.csv",
                    "solver_stats.csv",
                ):
                    self.assertTrue((day_dir / filename).exists(), filename)

            manifest = json.loads((Path(temp_dir) / "p1-review" / "manifest.json").read_text("utf-8"))
            self.assertEqual([item["test_day"] for item in manifest["episodes"]], [26, 27, 28, 29, 30])
            self.assertTrue(all(item["complete"] for item in manifest["episodes"]))

            with (Path(temp_dir) / "p1-review" / "evaluation_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual([int(row["test_day"]) for row in rows], [26, 27, 28, 29, 30])


if __name__ == "__main__":
    unittest.main()
