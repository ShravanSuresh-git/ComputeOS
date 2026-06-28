"""Common scheduling policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerDecision


class Scheduler(ABC):
    """Base class for dynamic compute scheduling policies.

    Implementations can be heuristic, supervised classifiers, contextual bandits,
    or reinforcement learning agents. The execution engine depends only on this
    API, keeping policy research isolated from model instrumentation.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset state before a new prompt, batch, or benchmark item."""

    @abstractmethod
    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        """Return a scheduling decision for the current execution context."""

    def observe(self, context: SchedulerContext, decision: SchedulerDecision) -> None:
        """Receive post-decision observations.

        Online or RL schedulers can override this method to update replay buffers,
        reward traces, or calibration statistics.
        """
        return None
