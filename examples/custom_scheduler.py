"""Minimal custom scheduler example.

Run this file directly to see how a new policy plugs into ComputeOS without
changing model code or model weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision


@dataclass
class LatencyBudgetScheduler(Scheduler):
    """Example scheduler that flags layers once a soft latency budget is exceeded."""

    soft_budget_ms: float = 10.0
    accumulated_latency_ms: float = 0.0

    def reset(self) -> None:
        self.accumulated_latency_ms = 0.0

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        layer = context.layer_telemetry
        if layer is None:
            return SchedulerDecision.record_only("waiting for layer telemetry")

        self.accumulated_latency_ms += layer.latency_ms
        if self.accumulated_latency_ms > self.soft_budget_ms:
            return SchedulerDecision(
                action=SchedulerAction.CONTINUE,
                layer_name=layer.layer_name,
                reason="soft latency budget exceeded",
                metadata={"accumulated_latency_ms": self.accumulated_latency_ms},
            )

        return SchedulerDecision(
            action=SchedulerAction.RECORD_ONLY,
            layer_name=layer.layer_name,
            reason="within latency budget",
            metadata={"accumulated_latency_ms": self.accumulated_latency_ms},
        )


def main() -> None:
    scheduler = LatencyBudgetScheduler(soft_budget_ms=2.5)
    print(scheduler)
    print("Register this class in `computeos.scheduling.registry` to use it from Hydra.")


if __name__ == "__main__":
    main()
