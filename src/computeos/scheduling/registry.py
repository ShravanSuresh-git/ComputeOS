"""Scheduler registry for policy construction."""

from __future__ import annotations

from collections.abc import Callable

from computeos.config.schema import SchedulerConfig
from computeos.scheduling.base import Scheduler
from computeos.scheduling.heuristic import HeuristicScheduler
from computeos.scheduling.pvs import (
    PVSCostWeights,
    PVSResourceBudgets,
    PVSValueWeights,
    PredictiveValueScheduler,
)

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
    registry.register("pvs", _pvs_factory)
    return registry


def _pvs_factory(config: SchedulerConfig) -> Scheduler:
    params = config.parameters
    budgets = dict(params.get("budgets", {})) if isinstance(params.get("budgets", {}), dict) else {}
    value_weights_param = params.get("value_weights", {})
    value_weights = dict(value_weights_param) if isinstance(value_weights_param, dict) else {}
    cost_weights_param = params.get("cost_weights", {})
    cost_weights = dict(cost_weights_param) if isinstance(cost_weights_param, dict) else {}
    return PredictiveValueScheduler(
        budgets=PVSResourceBudgets(
            max_latency_ms=float(budgets.get("max_latency_ms", 250.0)),
            max_memory_mb=float(budgets.get("max_memory_mb", 4096.0)),
            max_compute_units=float(budgets.get("max_compute_units", 512.0)),
            min_net_value=float(budgets.get("min_net_value", 0.0)),
        ),
        value_weights=PVSValueWeights(
            entropy=float(value_weights.get("entropy", 0.25)),
            uncertainty=float(value_weights.get("uncertainty", 0.30)),
            activation_norm=float(value_weights.get("activation_norm", 0.15)),
            layer_variance=float(value_weights.get("layer_variance", 0.15)),
            attention_entropy=float(value_weights.get("attention_entropy", 0.10)),
            decision_pressure=float(value_weights.get("decision_pressure", 0.05)),
        ),
        cost_weights=PVSCostWeights(
            compute=float(cost_weights.get("compute", 0.45)),
            latency=float(cost_weights.get("latency", 0.35)),
            memory=float(cost_weights.get("memory", 0.20)),
        ),
        utility_scale=float(params.get("utility_scale", 1.0)),
        cost_scale=float(params.get("cost_scale", 1.0)),
        smoothing=float(params.get("smoothing", 0.2)),
    )
