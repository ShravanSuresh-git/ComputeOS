from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from computeos.replay.trace_loader import ReplayTrace, TraceLoader
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.scheduling.pvs import PVSValueWeights, calibrate_weights
from computeos.telemetry.metrics import ActivationStats, LayerTelemetry, ModelTelemetry


def _trace(seed: float) -> ReplayTrace:
    telemetry = ModelTelemetry(model_name="calibration")
    for index in range(2):
        telemetry.layers.append(
            LayerTelemetry(
                layer_name=f"layer.{index}",
                layer_type="Linear",
                latency_ms=0.1,
                activation_stats=ActivationStats(
                    mean=0.0,
                    std=0.1,
                    min=-1.0,
                    max=1.0,
                    l2_norm=1.0,
                    numel=16,
                ),
            )
        )
        telemetry.scheduler_decisions.append(
            SchedulerDecision(
                action=SchedulerAction.CONTINUE,
                layer_name=f"layer.{index}",
                reason="calibration",
                metadata={
                    "features": {
                        "entropy": seed + index + 0.1,
                        "uncertainty": seed + 0.2,
                        "activation_norm": 0.3,
                        "layer_variance": 0.4,
                        "attention_entropy": 0.5,
                        "decision_pressure": 0.6,
                    }
                },
            )
        )
    telemetry.confidence_scores.append(0.95)
    telemetry.total_latency_ms = 0.2
    return TraceLoader().from_telemetry(telemetry)


class PVSCalibrationTests(unittest.TestCase):
    def test_calibrate_weights_returns_normalized_non_negative_weights(self) -> None:
        weights = calibrate_weights([_trace(0.0), _trace(1.0), _trace(2.0)])
        values = [
            weights.entropy,
            weights.uncertainty,
            weights.activation_norm,
            weights.layer_variance,
            weights.attention_entropy,
            weights.decision_pressure,
        ]

        self.assertIsInstance(weights, PVSValueWeights)
        self.assertAlmostEqual(sum(values), 1.0)
        self.assertTrue(all(value >= 0.0 for value in values))

    def test_calibrate_weights_raises_helpful_error_without_scipy(self) -> None:
        with patch.dict(sys.modules, {"scipy": None, "scipy.optimize": None}):
            with self.assertRaisesRegex(ImportError, "scipy"):
                calibrate_weights([_trace(0.0)])


if __name__ == "__main__":
    unittest.main()
