"""Attention-entropy baseline scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision


@dataclass
class EntropyScheduler(Scheduler):
    """Early-exit when attention entropy falls below a threshold."""

    threshold: float = 1.0

    def reset(self) -> None:
        """Entropy baseline is stateless."""

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        layer = context.layer_telemetry
        if layer is None or layer.attention_entropy is None:
            return SchedulerDecision.record_only("no attention entropy available")
        if layer.attention_entropy < self.threshold:
            return SchedulerDecision(
                action=SchedulerAction.EARLY_EXIT,
                layer_name=layer.layer_name,
                confidence=1.0 - (layer.attention_entropy / self.threshold),
                reason=(
                    f"attention entropy {layer.attention_entropy:.4f} "
                    f"below threshold {self.threshold}"
                ),
            )
        return SchedulerDecision(
            action=SchedulerAction.CONTINUE,
            layer_name=layer.layer_name,
            reason="attention entropy above threshold",
        )
