"""Scheduler decision value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SchedulerAction(StrEnum):
    """Actions a scheduler can request from the execution engine."""

    CONTINUE = "continue"
    EARLY_EXIT = "early_exit"
    SKIP_LAYER = "skip_layer"
    ADJUST_CACHE = "adjust_cache"
    RECORD_ONLY = "record_only"


@dataclass(frozen=True)
class SchedulerDecision:
    """A decision emitted by a scheduling policy at an execution boundary."""

    action: SchedulerAction = SchedulerAction.CONTINUE
    layer_name: str | None = None
    confidence: float | None = None
    reason: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def record_only(cls, reason: str = "observability pass") -> "SchedulerDecision":
        """Create a no-op decision used by baseline policies."""

        return cls(action=SchedulerAction.RECORD_ONLY, reason=reason)
