from __future__ import annotations

import unittest

from computeos.config.schema import SchedulerConfig
from computeos.scheduling.decision import SchedulerAction
from computeos.scheduling.registry import default_scheduler_registry
from computeos.telemetry.metrics import LayerTelemetry, ModelTelemetry
from computeos.scheduling.context import SchedulerContext


class SchedulerTests(unittest.TestCase):
    def test_registry_builds_heuristic_scheduler(self) -> None:
        scheduler = default_scheduler_registry().create(
            SchedulerConfig(name="heuristic", parameters={"entropy_threshold": 2.0})
        )
        layer = LayerTelemetry(
            layer_name="transformer.h.0",
            layer_type="GPT2Block",
            latency_ms=1.0,
            attention_entropy=1.0,
        )
        decision = scheduler.decide(
            SchedulerContext(
                step_index=0,
                layer_name=layer.layer_name,
                layer_telemetry=layer,
                model_telemetry=ModelTelemetry(model_name="tiny"),
            )
        )
        self.assertEqual(decision.action, SchedulerAction.CONTINUE)
        self.assertEqual(decision.layer_name, "transformer.h.0")

    def test_unknown_scheduler_has_helpful_error(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown scheduler"):
            default_scheduler_registry().create(SchedulerConfig(name="missing"))


if __name__ == "__main__":
    unittest.main()
