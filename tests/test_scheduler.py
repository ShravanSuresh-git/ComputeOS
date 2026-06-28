from __future__ import annotations

import unittest

from computeos.config.schema import SchedulerConfig
from computeos.scheduling.confidence import ConfidenceScheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction
from computeos.scheduling.entropy import EntropyScheduler
from computeos.scheduling.random_scheduler import RandomScheduler
from computeos.scheduling.registry import default_scheduler_registry
from computeos.telemetry.metrics import LayerTelemetry, ModelTelemetry


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

    def test_entropy_scheduler_emits_early_exit(self) -> None:
        scheduler = EntropyScheduler(threshold=1.0)
        layer = LayerTelemetry(
            layer_name="transformer.h.0",
            layer_type="GPT2Block",
            latency_ms=1.0,
            attention_entropy=0.01,
        )

        decision = scheduler.decide(
            SchedulerContext(
                step_index=0,
                layer_name=layer.layer_name,
                layer_telemetry=layer,
                model_telemetry=ModelTelemetry(model_name="test"),
            )
        )

        self.assertEqual(decision.action, SchedulerAction.EARLY_EXIT)

    def test_confidence_scheduler_emits_early_exit(self) -> None:
        scheduler = ConfidenceScheduler(threshold=0.95)
        telemetry = ModelTelemetry(model_name="test")
        telemetry.confidence_scores.append(0.99)

        decision = scheduler.decide(
            SchedulerContext(
                step_index=0,
                layer_name=None,
                layer_telemetry=None,
                model_telemetry=telemetry,
            )
        )

        self.assertEqual(decision.action, SchedulerAction.EARLY_EXIT)

    def test_random_scheduler_emits_early_exit_when_probability_is_one(self) -> None:
        scheduler = RandomScheduler(exit_prob=1.0, seed=42)
        scheduler.reset()
        context = SchedulerContext(
            step_index=0,
            layer_name=None,
            layer_telemetry=None,
            model_telemetry=ModelTelemetry(model_name="test"),
        )

        decisions = [scheduler.decide(context) for _ in range(1000)]

        self.assertTrue(
            all(decision.action == SchedulerAction.EARLY_EXIT for decision in decisions)
        )


if __name__ == "__main__":
    unittest.main()
