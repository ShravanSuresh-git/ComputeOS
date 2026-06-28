"""Seeded random baseline scheduler."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision


@dataclass
class RandomScheduler(Scheduler):
    """Randomly request early exit with a fixed probability."""

    exit_prob: float = 0.1
    seed: int = 42
    _rng: random.Random = field(
        default_factory=lambda: random.Random(42),
        init=False,
        repr=False,
    )

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        if self._rng.random() < self.exit_prob:
            return SchedulerDecision(
                action=SchedulerAction.EARLY_EXIT,
                layer_name=context.layer_name,
                reason=f"random exit (p={self.exit_prob})",
            )
        return SchedulerDecision(
            action=SchedulerAction.CONTINUE,
            layer_name=context.layer_name,
            reason="random continue",
        )
