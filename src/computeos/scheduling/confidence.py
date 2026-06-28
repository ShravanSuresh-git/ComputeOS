"""Token-confidence baseline scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision


@dataclass
class ConfidenceScheduler(Scheduler):
    """Early-exit when the latest token confidence exceeds a threshold."""

    threshold: float = 0.95

    def reset(self) -> None:
        """Confidence baseline is stateless."""

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        scores = context.model_telemetry.confidence_scores
        if not scores:
            return SchedulerDecision.record_only("no confidence scores yet")
        latest = min(max(scores[-1], 0.0), 1.0)
        if latest >= self.threshold:
            return SchedulerDecision(
                action=SchedulerAction.EARLY_EXIT,
                layer_name=context.layer_name,
                confidence=latest,
                reason=f"token confidence {latest:.4f} exceeds threshold {self.threshold}",
            )
        return SchedulerDecision(
            action=SchedulerAction.CONTINUE,
            layer_name=context.layer_name,
            reason="confidence below threshold",
        )
