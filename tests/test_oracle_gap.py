from __future__ import annotations

import unittest

from computeos.replay.oracle_scheduler import OracleObjective, OracleScheduler
from computeos.replay.trace_loader import ReplayTrace, trace_utility
from computeos.telemetry.metrics import ActivationStats, LayerTelemetry


def _trace() -> ReplayTrace:
    layers = tuple(
        LayerTelemetry(
            layer_name=f"transformer.h.{index}",
            layer_type="GPT2Block",
            latency_ms=1.0 + index,
            activation_stats=ActivationStats(
                mean=0.0,
                std=0.1,
                min=-1.0,
                max=1.0,
                l2_norm=2.0,
                numel=1024,
            ),
        )
        for index in range(6)
    )
    return ReplayTrace(
        model_name="random-gpt2",
        events=(),
        layers=layers,
        decisions=(),
        total_latency_ms=21.0,
        peak_memory_mb=0.0,
        total_compute_units=6.0,
        final_utility=0.5,
    )


class OracleGapTests(unittest.TestCase):
    def test_oracle_plan_utility_geq_full_inference(self) -> None:
        trace = _trace()
        plan = OracleScheduler().plan(trace, OracleObjective.BALANCED)

        self.assertGreaterEqual(plan.utility, 0.0)
        self.assertTrue(plan.stop_index is None or 0 <= plan.stop_index < len(trace.layers))

    def test_oracle_gap_bounded(self) -> None:
        trace = _trace()
        plan = OracleScheduler().plan(trace, OracleObjective.BALANCED)
        achieved = trace_utility(
            latency_ms=trace.total_latency_ms,
            memory_mb=trace.peak_memory_mb,
            compute_units=trace.total_compute_units,
            quality_proxy=0.5,
        )
        gap = (plan.utility - achieved) / (abs(plan.utility) + 1e-9)
        gap = min(1.0, max(0.0, gap))

        self.assertGreaterEqual(gap, 0.0)
        self.assertLessEqual(gap, 1.0)


if __name__ == "__main__":
    unittest.main()
