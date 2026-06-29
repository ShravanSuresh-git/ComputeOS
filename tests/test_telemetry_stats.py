from __future__ import annotations

import math
import unittest

import torch

from computeos.telemetry.metrics import LayerTelemetry, ModelTelemetry
from computeos.telemetry.stats import activation_stats, attention_entropy


class TelemetryStatsTests(unittest.TestCase):
    def test_activation_stats_detaches_tensor_summary(self) -> None:
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        stats = activation_stats(tensor)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.numel, 4)
        self.assertAlmostEqual(stats.mean, 2.5)
        self.assertAlmostEqual(stats.min, 1.0)
        self.assertAlmostEqual(stats.max, 4.0)

    def test_attention_entropy_from_probability_tensor(self) -> None:
        probs = torch.tensor([[[[0.5, 0.5], [1.0, 1e-12]]]])
        entropy = attention_entropy(probs)
        self.assertIsNotNone(entropy)
        assert entropy is not None
        self.assertTrue(math.isfinite(entropy))
        self.assertGreater(entropy, 0.0)

    def test_model_telemetry_has_unique_request_ids(self) -> None:
        first = ModelTelemetry(model_name="test")
        second = ModelTelemetry(model_name="test")

        self.assertIsInstance(first.request_id, str)
        self.assertIsInstance(second.request_id, str)
        self.assertNotEqual(first.request_id, second.request_id)

    def test_attention_entropy_availability_is_false_when_none(self) -> None:
        layer = LayerTelemetry(
            layer_name="layer",
            layer_type="Linear",
            latency_ms=1.0,
            attention_entropy=None,
        )

        self.assertFalse(layer.attention_entropy_available)


if __name__ == "__main__":
    unittest.main()
