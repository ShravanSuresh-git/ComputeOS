"""Scheduler registry for policy construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from computeos.config.schema import SchedulerConfig
from computeos.scheduling.base import Scheduler
from computeos.scheduling.confidence import ConfidenceScheduler
from computeos.scheduling.entropy import EntropyScheduler
from computeos.scheduling.heuristic import HeuristicScheduler
from computeos.scheduling.pvs import (
    PredictiveValueScheduler,
    PVSCostWeights,
    PVSResourceBudgets,
    PVSValueWeights,
)
from computeos.scheduling.random_scheduler import RandomScheduler

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
        confidence_threshold=_float_param(params, "confidence_threshold", 0.85),
        entropy_threshold=_float_param(params, "entropy_threshold", 1.5),
    )


def default_scheduler_registry() -> SchedulerRegistry:
    registry = SchedulerRegistry()
    registry.register("confidence", _confidence_factory)
    registry.register("entropy", _entropy_factory)
    registry.register("heuristic", _heuristic_factory)
    registry.register("pvs", _pvs_factory)
    registry.register("random", _random_factory)
    return registry


def _entropy_factory(config: SchedulerConfig) -> Scheduler:
    return EntropyScheduler(threshold=_float_param(config.parameters, "threshold", 1.0))


def _confidence_factory(config: SchedulerConfig) -> Scheduler:
    return ConfidenceScheduler(threshold=_float_param(config.parameters, "threshold", 0.95))


def _random_factory(config: SchedulerConfig) -> Scheduler:
    return RandomScheduler(
        exit_prob=_float_param(config.parameters, "exit_prob", 0.1),
        seed=_int_param(config.parameters, "seed", 42),
    )


def _pvs_factory(config: SchedulerConfig) -> Scheduler:
    params = config.parameters
    budgets = _mapping_param(params, "budgets")
    value_weights = _mapping_param(params, "value_weights")
    cost_weights = _mapping_param(params, "cost_weights")
    return PredictiveValueScheduler(
        budgets=PVSResourceBudgets(
            max_latency_ms=_float_param(budgets, "max_latency_ms", 250.0),
            max_memory_mb=_float_param(budgets, "max_memory_mb", 4096.0),
            max_compute_units=_float_param(budgets, "max_compute_units", 512.0),
            min_net_value=_float_param(budgets, "min_net_value", 0.0),
        ),
        value_weights=PVSValueWeights(
            entropy=_float_param(value_weights, "entropy", 0.25),
            uncertainty=_float_param(value_weights, "uncertainty", 0.30),
            activation_norm=_float_param(value_weights, "activation_norm", 0.15),
            layer_variance=_float_param(value_weights, "layer_variance", 0.15),
            attention_entropy=_float_param(value_weights, "attention_entropy", 0.10),
            decision_pressure=_float_param(value_weights, "decision_pressure", 0.05),
        ),
        cost_weights=PVSCostWeights(
            compute=_float_param(cost_weights, "compute", 0.45),
            latency=_float_param(cost_weights, "latency", 0.35),
            memory=_float_param(cost_weights, "memory", 0.20),
        ),
        utility_scale=_float_param(params, "utility_scale", 1.0),
        cost_scale=_float_param(params, "cost_scale", 1.0),
        smoothing=_float_param(params, "smoothing", 0.2),
    )


def _float_param(params: Mapping[str, object], key: str, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def _int_param(params: Mapping[str, object], key: str, default: int) -> int:
    value = params.get(key, default)
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def _mapping_param(params: Mapping[str, object], key: str) -> dict[str, object]:
    value = params.get(key, {})
    if isinstance(value, Mapping):
        return dict(value)
    return {}
