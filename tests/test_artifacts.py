from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from computeos.experiments.artifacts import ArtifactStore
from computeos.experiments.comparison import ComparisonReport
from computeos.replay.regret import SchedulerRegret
from computeos.telemetry.metrics import ModelTelemetry


class ArtifactStoreTests(unittest.TestCase):
    def test_artifact_store_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(Path(temp_dir), run_id="run")
            telemetry = ModelTelemetry(model_name="tiny")
            report = ComparisonReport(
                rows=[
                    {
                        "scheduler": "heuristic",
                        "prompt": "hello",
                        "score": 1.0,
                    }
                ],
                regret_by_scheduler={
                    "heuristic": SchedulerRegret((), 0.0, 0.0, 0.0, 0.0),
                },
            )

            config_path = store.snapshot_config({"name": "test"})
            env_path = store.snapshot_env()
            telemetry_path = store.snapshot_telemetry(telemetry)
            report_path = store.snapshot_report(report)

            self.assertEqual(json.loads(config_path.read_text(encoding="utf-8"))["name"], "test")
            self.assertIn("python_version", json.loads(env_path.read_text(encoding="utf-8")))
            telemetry_lines = telemetry_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(telemetry_lines[0])["model_name"], "tiny")
            self.assertIn("regret", json.loads(report_path.read_text(encoding="utf-8")))
            with (store.run_dir / "report.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["scheduler"], "heuristic")


if __name__ == "__main__":
    unittest.main()
