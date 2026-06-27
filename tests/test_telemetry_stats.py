from __future__ import annotations

import math
import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
