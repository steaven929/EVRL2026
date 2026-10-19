import json
from pathlib import Path
import tempfile
import unittest

from core.run_provenance import write_run_provenance


class RunProvenanceTests(unittest.TestCase):
    def test_writes_effective_config_git_version_and_source_manifest(self):
        config = {
            "experiment_id": "E2",
            "experiment_variant": "full",
            "seed": 1,
            "train_day_indices": [1, 2, 3],
            "path_value": Path("data/SIP"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            written = write_run_provenance(temp_dir, config)
            output = Path(temp_dir)
            self.assertEqual(
                set(written),
                {"effective_config", "git_version", "source_manifest"},
            )
            effective = json.loads((output / "effective_config.json").read_text(encoding="utf-8"))
            git_version = json.loads((output / "git_version.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(effective["experiment_id"], "E2")
        self.assertEqual(effective["train_day_indices"], [1, 2, 3])
        self.assertEqual(effective["path_value"], str(Path("data/SIP")))
        self.assertRegex(git_version["commit"], r"^[0-9a-f]{40}$")
        self.assertIn("branch", git_version)
        self.assertIn("working_tree_dirty", git_version)
        self.assertEqual(git_version["source_manifest_sha256"], manifest["manifest_sha256"])
        self.assertGreater(manifest["source_file_count"], 0)
        self.assertIn("core/util.py", manifest["files"])


if __name__ == "__main__":
    unittest.main()
