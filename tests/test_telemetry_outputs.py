from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.telemetry.loggers import CsvTelemetryLogger, JsonTelemetryLogger
from computeos.telemetry.metrics import ActivationStats, LayerTelemetry, ModelTelemetry
from computeos.telemetry.reports import render_telemetry_report


def _sample_telemetry() -> ModelTelemetry:
    telemetry = ModelTelemetry(model_name="tiny")
    telemetry.layers.append(
        LayerTelemetry(
            layer_name="transformer.h.0",
            layer_type="GPT2Block",
            latency_ms=1.25,
            activation_stats=ActivationStats(
                mean=0.1,
                std=0.2,
                min=-0.3,
                max=0.4,
                l2_norm=1.5,
                numel=8,
            ),
            attention_entropy=0.9,
            process_rss_bytes=1024,
        )
    )
    telemetry.scheduler_decisions.append(
        SchedulerDecision(
            action=SchedulerAction.RECORD_ONLY,
            layer_name="transformer.h.0",
            reason="test",
        )
    )
    telemetry.confidence_scores.append(0.7)
    telemetry.total_latency_ms = 2.5
    telemetry.peak_process_rss_bytes = 2048
    return telemetry


class TelemetryOutputTests(unittest.TestCase):
    def test_json_export_writes_model_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "telemetry.json"
            logger = JsonTelemetryLogger(path)
            logger.log(_sample_telemetry())
            logger.close()

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["model_name"], "tiny")
            self.assertEqual(payload[0]["layers"][0]["layer_name"], "transformer.h.0")

    def test_csv_export_writes_layer_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "layers.csv"
            logger = CsvTelemetryLogger(path)
            logger.log(_sample_telemetry())
            logger.close()

            with path.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["model_name"], "tiny")
            self.assertEqual(rows[0]["layer_name"], "transformer.h.0")

    def test_render_report_contains_summary_and_layers(self) -> None:
        report = render_telemetry_report(_sample_telemetry())
        self.assertIn("ComputeOS Telemetry Summary", report)
        self.assertIn("transformer.h.0", report)


if __name__ == "__main__":
    unittest.main()
