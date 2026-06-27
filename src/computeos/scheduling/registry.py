"""Scheduler registry for policy construction."""

from __future__ import annotations

from collections.abc import Callable

from computeos.config.schema import SchedulerConfig
from computeos.scheduling.base import Scheduler
from computeos.scheduling.heuristic import HeuristicScheduler

SchedulerFactory = Callable[[SchedulerConfig], Scheduler]


class SchedulerRegistry:
    """Small explicit registry that keeps policy discovery testable."""

    def __init__(self) -> None:
        self._factories: dict[str, SchedulerFactory] = {}

    def register(self, name: str, factory: SchedulerFactory) -> None:
        if not name:
            raise ValueError("Scheduler name must be non-empty.")
        self._factories[name] = factory

    def create(self, config: SchedulerConfig) -> Scheduler:
        try:
            factory = self._factories[config.name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown scheduler '{config.name}'. Available: {available}") from exc
        return factory(config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _heuristic_factory(config: SchedulerConfig) -> Scheduler:
    params = config.parameters
    return HeuristicScheduler(
        confidence_threshold=float(params.get("confidence_threshold", 0.85)),
        entropy_threshold=float(params.get("entropy_threshold", 1.5)),
    )


def default_scheduler_registry() -> SchedulerRegistry:
    registry = SchedulerRegistry()
    registry.register("heuristic", _heuristic_factory)
    return registry
