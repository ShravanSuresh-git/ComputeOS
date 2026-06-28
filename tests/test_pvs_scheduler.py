from __future__ import annotations

import unittest

from computeos.config.schema import SchedulerConfig
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction
from computeos.scheduling.pvs import (
    PVSResourceBudgets,
    PredictiveValueScheduler,
)
from computeos.scheduling.registry import default_scheduler_registry
from computeos.telemetry.collector import TelemetryCollector
from computeos.telemetry.metrics import ActivationStats, LayerTelemetry, ModelTelemetry
from computeos.visualization.pvs import extract_pvs_trace


def _layer(
    latency_ms: float = 1.0,
    attention_entropy: float | None = 2.0,
    l2_norm: float = 8.0,
) -> LayerTelemetry:
    return LayerTelemetry(
        layer_name="transformer.h.0",
        layer_type="GPT2Block",
        latency_ms=latency_ms,
        activation_stats=ActivationStats(
            mean=0.0,
            std=0.5,
            min=-1.0,
            max=1.0,
            l2_norm=l2_norm,
            numel=2048,
        ),
        attention_entropy=attention_entropy,
        process_rss_bytes=128 * 1024 * 1024,
    )


def _context(layer: LayerTelemetry, telemetry: ModelTelemetry | None = None) -> SchedulerContext:
    model_telemetry = telemetry or ModelTelemetry(model_name="tiny")
    return SchedulerContext(
        step_index=len(model_telemetry.layers),
        layer_name=layer.layer_name,
        layer_telemetry=layer,
        model_telemetry=model_telemetry,
    )


class PredictiveValueSchedulerTests(unittest.TestCase):
    def test_registry_builds_pvs_scheduler(self) -> None:
        scheduler = default_scheduler_registry().create(
            SchedulerConfig(
                name="pvs",
                parameters={"budgets": {"max_latency_ms": 100.0}},
            )
        )
        self.assertIsInstance(scheduler, PredictiveValueScheduler)

    def test_pvs_continues_when_marginal_value_exceeds_cost(self) -> None:
        scheduler = PredictiveValueScheduler(
            budgets=PVSResourceBudgets(
                max_latency_ms=1000.0,
                max_memory_mb=4096.0,
                max_compute_units=1000.0,
                min_net_value=0.0,
            )
        )
        decision = scheduler.decide(_context(_layer()))

        self.assertEqual(decision.action, SchedulerAction.CONTINUE)
        self.assertEqual(decision.metadata["algorithm"], "predictive_value_scheduling")
        self.assertGreater(
            decision.metadata["prediction"]["expected_net_value"],
            0.0,
        )

    def test_pvs_stops_when_latency_budget_is_exhausted(self) -> None:
        scheduler = PredictiveValueScheduler(
            budgets=PVSResourceBudgets(
                max_latency_ms=0.5,
                max_memory_mb=4096.0,
                max_compute_units=1000.0,
            )
        )
        decision = scheduler.decide(_context(_layer(latency_ms=1.0)))

        self.assertEqual(decision.action, SchedulerAction.EARLY_EXIT)
        self.assertTrue(decision.metadata["stopping_event"])
        self.assertEqual(decision.reason, "latency budget exhausted")

    def test_pvs_replay_and_telemetry_trace_extraction(self) -> None:
        scheduler = PredictiveValueScheduler()
        collector = TelemetryCollector(model_name="tiny")
        layer = _layer()
        collector.record_layer(layer)
        decision = scheduler.decide(_context(layer, collector.model_telemetry))
        collector.record_decision(decision)

        replay = scheduler.replay()
        extracted = extract_pvs_trace(collector.model_telemetry)

        self.assertEqual(len(replay), 1)
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["layer_name"], "transformer.h.0")
        self.assertIn("expected_improvement", extracted[0]["prediction"])


if __name__ == "__main__":
    unittest.main()
