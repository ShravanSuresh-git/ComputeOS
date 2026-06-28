"""Counterfactual Runtime Intelligence engine."""

from __future__ import annotations

from dataclasses import dataclass

from computeos.replay.metrics import (
    ReplayMetrics,
    budget_efficiency,
    decision_stability,
    expected_information_gain,
    utility_per_flop,
    utility_per_joule,
)
from computeos.replay.oracle_scheduler import OracleObjective, OracleScheduler
from computeos.replay.regret import SchedulerRegret, compute_regret
from computeos.replay.scenario import CounterfactualScenario, ScenarioType
from computeos.replay.trace_loader import (
    ReplayTrace,
    estimate_compute_units,
    first_stopping_index,
    layer_memory_mb,
    trace_utility,
)
from computeos.telemetry.metrics import LayerTelemetry


@dataclass(frozen=True)
class CounterfactualResult:
    """Predicted outcome for one counterfactual scenario."""

    scenario: CounterfactualScenario
    predicted_utility: float
    predicted_latency_ms: float
    predicted_compute_units: float
    predicted_memory_mb: float
    predicted_quality_proxy: float
    metrics: ReplayMetrics
    regret: SchedulerRegret
    explanation: str


class CounterfactualEngine:
    """Evaluate alternative scheduling decisions from completed traces.

    The engine reuses observed telemetry whenever possible. It does not claim to
    know unobserved model outputs; estimates are explicit proxies designed to be
    replaced by future predictive models.
    """

    def __init__(self, oracle: OracleScheduler | None = None) -> None:
        self._oracle = oracle or OracleScheduler()

    def evaluate(
        self,
        trace: ReplayTrace,
        scenario: CounterfactualScenario,
        objective: OracleObjective = OracleObjective.MAXIMIZE_UTILITY,
    ) -> CounterfactualResult:
        latency, compute, memory, quality, explanation = self._simulate(trace, scenario)
        utility = trace_utility(
            latency_ms=latency,
            memory_mb=memory,
            compute_units=compute,
            quality_proxy=quality,
        )
        oracle = self._oracle.plan(trace, objective=objective)
        online_utilities = _decision_utilities(trace)
        oracle_utilities = [decision.utility for decision in oracle.decisions]
        regret = compute_regret(oracle_utilities, online_utilities)
        actions = [str(decision.action) for decision in trace.decisions]
        oracle_actions = [decision.action for decision in oracle.decisions]
        metrics = ReplayMetrics(
            utility=utility,
            latency_ms=latency,
            compute_units=compute,
            memory_mb=memory,
            quality_proxy=quality,
            budget_efficiency=budget_efficiency(utility, compute, latency),
            expected_information_gain=expected_information_gain(quality, _trace_quality(trace)),
            utility_per_flop=utility_per_flop(utility, compute),
            utility_per_joule=utility_per_joule(utility),
            counterfactual_improvement=utility - trace.final_utility,
            decision_stability=decision_stability(actions, oracle_actions),
            stopping_accuracy=_stopping_accuracy(trace, oracle.stop_index),
            oracle_gap=oracle.utility - utility,
        )
        return CounterfactualResult(
            scenario=scenario,
            predicted_utility=utility,
            predicted_latency_ms=latency,
            predicted_compute_units=compute,
            predicted_memory_mb=memory,
            predicted_quality_proxy=quality,
            metrics=metrics,
            regret=regret,
            explanation=explanation,
        )

    def evaluate_many(
        self,
        trace: ReplayTrace,
        scenarios: list[CounterfactualScenario],
        objective: OracleObjective = OracleObjective.MAXIMIZE_UTILITY,
    ) -> list[CounterfactualResult]:
        return [self.evaluate(trace, scenario, objective=objective) for scenario in scenarios]

    def _simulate(
        self,
        trace: ReplayTrace,
        scenario: CounterfactualScenario,
    ) -> tuple[float, float, float, float, str]:
        if scenario.scenario_type == ScenarioType.CONTINUE_EXTRA_LAYERS:
            return _continue_extra_layers(trace, scenario.extra_layers)
        if scenario.scenario_type == ScenarioType.INCREASE_REASONING_BUDGET:
            return _budget_scale(trace, scale=1.25, reason="increased reasoning budget")
        if scenario.scenario_type == ScenarioType.DECREASE_REASONING_BUDGET:
            return _budget_scale(trace, scale=0.75, reason="decreased reasoning budget")
        if scenario.scenario_type == ScenarioType.CHANGE_LATENCY_BUDGET:
            return _apply_latency_budget(trace, scenario.latency_budget_ms)
        if scenario.scenario_type == ScenarioType.CHANGE_COMPUTE_BUDGET:
            return _apply_compute_budget(trace, scenario.compute_budget_units)
        if scenario.scenario_type == ScenarioType.CHANGE_MEMORY_BUDGET:
            return _apply_memory_budget(trace, scenario.memory_budget_mb)
        if scenario.scenario_type == ScenarioType.REPLACE_SCHEDULER:
            return _replace_scheduler(trace, scenario.scheduler_name)
        if scenario.scenario_type == ScenarioType.CHANGE_STOPPING_CRITERION:
            return _change_stopping_criterion(trace, scenario.stopping_threshold)
        return (
            trace.total_latency_ms,
            trace.total_compute_units,
            trace.peak_memory_mb,
            _trace_quality(trace),
            "no-op scenario",
        )


def _continue_extra_layers(
    trace: ReplayTrace,
    extra_layers: int,
) -> tuple[float, float, float, float, str]:
    stop_index = first_stopping_index(trace.decisions)
    if stop_index is None:
        return (
            trace.total_latency_ms,
            trace.total_compute_units,
            trace.peak_memory_mb,
            _trace_quality(trace),
            "trace already contains full observed execution",
        )
    target = min(len(trace.layers), stop_index + 1 + max(0, extra_layers))
    layers = trace.layers[:target]
    latency = sum(max(0.0, layer.latency_ms) for layer in layers)
    compute = sum(estimate_compute_units(layer) for layer in layers)
    memory = max((layer_memory_mb(layer) or 0.0 for layer in layers), default=0.0)
    quality = _quality_for_fraction(trace, target / max(1, len(trace.layers)))
    return latency, compute, memory, quality, f"continued {extra_layers} extra layers"


def _budget_scale(
    trace: ReplayTrace,
    scale: float,
    reason: str,
) -> tuple[float, float, float, float, str]:
    target = max(1, min(len(trace.layers), int(round(len(trace.layers) * scale))))
    layers = trace.layers[:target]
    latency = sum(max(0.0, layer.latency_ms) for layer in layers)
    compute = sum(estimate_compute_units(layer) for layer in layers)
    memory = max((layer_memory_mb(layer) or 0.0 for layer in layers), default=trace.peak_memory_mb)
    quality = _quality_for_fraction(trace, target / max(1, len(trace.layers)))
    return latency, compute, memory, quality, reason


def _apply_latency_budget(
    trace: ReplayTrace,
    latency_budget_ms: float | None,
) -> tuple[float, float, float, float, str]:
    if latency_budget_ms is None:
        return _identity(trace, "latency budget unchanged")
    selected = []
    elapsed = 0.0
    for layer in trace.layers:
        if elapsed + layer.latency_ms > latency_budget_ms:
            break
        selected.append(layer)
        elapsed += layer.latency_ms
    return _from_layers(trace, selected, f"latency budget set to {latency_budget_ms:.3f} ms")


def _apply_compute_budget(
    trace: ReplayTrace,
    compute_budget_units: float | None,
) -> tuple[float, float, float, float, str]:
    if compute_budget_units is None:
        return _identity(trace, "compute budget unchanged")
    selected = []
    spent = 0.0
    for layer in trace.layers:
        cost = estimate_compute_units(layer)
        if spent + cost > compute_budget_units:
            break
        selected.append(layer)
        spent += cost
    return _from_layers(trace, selected, f"compute budget set to {compute_budget_units:.3f}")


def _apply_memory_budget(
    trace: ReplayTrace,
    memory_budget_mb: float | None,
) -> tuple[float, float, float, float, str]:
    if memory_budget_mb is None:
        return _identity(trace, "memory budget unchanged")
    selected = []
    peak = 0.0
    for layer in trace.layers:
        peak = max(peak, layer_memory_mb(layer) or 0.0)
        if peak > memory_budget_mb:
            break
        selected.append(layer)
    return _from_layers(trace, selected, f"memory budget set to {memory_budget_mb:.3f} MB")


def _replace_scheduler(
    trace: ReplayTrace,
    scheduler_name: str | None,
) -> tuple[float, float, float, float, str]:
    if scheduler_name in {None, "static"}:
        return _identity(trace, "static scheduler executes all observed layers")
    if scheduler_name == "random":
        target = max(1, len(trace.layers) // 2)
        return _from_layers(trace, list(trace.layers[:target]), "deterministic random proxy")
    if scheduler_name == "confidence":
        target = max(1, int(len(trace.layers) * 0.75))
        return _from_layers(trace, list(trace.layers[:target]), "confidence scheduler proxy")
    if scheduler_name == "entropy":
        target = max(1, int(len(trace.layers) * 0.80))
        return _from_layers(trace, list(trace.layers[:target]), "entropy scheduler proxy")
    if scheduler_name == "pvs":
        stop = first_stopping_index(trace.decisions)
        if stop is None:
            return _identity(trace, "pvs trace has no stopping decision")
        return _from_layers(trace, list(trace.layers[: stop + 1]), "pvs stopping replay")
    return _identity(trace, f"unknown scheduler proxy: {scheduler_name}")


def _change_stopping_criterion(
    trace: ReplayTrace,
    stopping_threshold: float | None,
) -> tuple[float, float, float, float, str]:
    threshold = 0.0 if stopping_threshold is None else stopping_threshold
    for index, decision in enumerate(trace.decisions):
        prediction = decision.metadata.get("prediction")
        if isinstance(prediction, dict):
            net_value = float(prediction.get("expected_net_value", 0.0))
            if net_value <= threshold:
                return _from_layers(
                    trace,
                    list(trace.layers[: index + 1]),
                    f"stopping threshold set to {threshold:.3f}",
                )
    return _identity(trace, f"stopping threshold set to {threshold:.3f}")


def _identity(trace: ReplayTrace, reason: str) -> tuple[float, float, float, float, str]:
    return (
        trace.total_latency_ms,
        trace.total_compute_units,
        trace.peak_memory_mb,
        _trace_quality(trace),
        reason,
    )


def _from_layers(
    trace: ReplayTrace,
    layers: list[LayerTelemetry],
    reason: str,
) -> tuple[float, float, float, float, str]:
    latency = sum(max(0.0, layer.latency_ms) for layer in layers)
    compute = sum(estimate_compute_units(layer) for layer in layers)
    memory = max((layer_memory_mb(layer) or 0.0 for layer in layers), default=0.0)
    quality = _quality_for_fraction(trace, len(layers) / max(1, len(trace.layers)))
    return latency, compute, memory, quality, reason


def _trace_quality(trace: ReplayTrace) -> float:
    return max(0.0, trace.final_utility + 0.001 * trace.total_latency_ms)


def _quality_for_fraction(trace: ReplayTrace, fraction: float) -> float:
    fraction = min(1.0, max(0.0, fraction))
    return _trace_quality(trace) * fraction


def _decision_utilities(trace: ReplayTrace) -> list[float]:
    utilities = [event.utility for event in trace.decision_events() if event.utility is not None]
    if utilities:
        return [float(utility) for utility in utilities]
    return [trace.final_utility]


def _stopping_accuracy(trace: ReplayTrace, oracle_stop_index: int | None) -> float:
    actual_stop = first_stopping_index(trace.decisions)
    if actual_stop == oracle_stop_index:
        return 1.0
    if actual_stop is None or oracle_stop_index is None:
        return 0.0
    distance = abs(actual_stop - oracle_stop_index)
    return max(0.0, 1.0 - distance / max(1, len(trace.layers)))
