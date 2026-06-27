"""Baseline heuristic scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision


@dataclass
class HeuristicScheduler(Scheduler):
    """Simple policy that records telemetry and flags low-uncertainty states.

    This scheduler intentionally avoids altering model execution. It demonstrates
    the extension contract while providing meaningful decision logs for baselines.
    """

    confidence_threshold: float = 0.85
    entropy_threshold: float = 1.5

    def reset(self) -> None:
        """Heuristic policy is stateless."""

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        layer = context.layer_telemetry
        if layer is None:
            return SchedulerDecision.record_only("no layer telemetry yet")

        entropy = layer.attention_entropy
        if entropy is not None and entropy < self.entropy_threshold:
            return SchedulerDecision(
                action=SchedulerAction.CONTINUE,
                layer_name=layer.layer_name,
                confidence=min(1.0, self.confidence_threshold),
                reason="attention entropy below heuristic threshold",
                metadata={"attention_entropy": entropy},
            )

        return SchedulerDecision(
            action=SchedulerAction.RECORD_ONLY,
            layer_name=layer.layer_name,
            reason="baseline telemetry collection",
        )
