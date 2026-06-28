"""Offline oracle scheduler for completed traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from computeos.replay.trace_loader import ReplayTrace, estimate_compute_units, layer_memory_mb
from computeos.scheduling.decision import SchedulerAction


class OracleObjective(StrEnum):
    """Offline optimization objectives supported by the oracle scheduler."""

    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_LATENCY = "minimize_latency"
    MINIMIZE_FLOPS = "minimize_flops"
    MINIMIZE_MEMORY = "minimize_memory"
    MAXIMIZE_UTILITY = "maximize_utility"
    BALANCED = "balanced"


@dataclass(frozen=True)
class OracleDecision:
    """Offline oracle decision for one layer boundary."""

    index: int
    layer_name: str
    action: str
    utility: float
    cumulative_latency_ms: float
    cumulative_compute_units: float
    peak_memory_mb: float


@dataclass(frozen=True)
class OraclePlan:
    """Full offline oracle plan."""

    objective: OracleObjective
    decisions: tuple[OracleDecision, ...]
    utility: float
    stop_index: int | None


@dataclass(frozen=True)
class OracleConstraints:
    """Optional oracle resource constraints."""

    latency_budget_ms: float | None = None
    compute_budget_units: float | None = None
    memory_budget_mb: float | None = None


class OracleScheduler:
    """Offline-only oracle that optimizes decisions from a completed trace.

    This class intentionally does not implement the live `Scheduler` interface.
    It has access to the full trace and must never be used during inference.
    """

    def plan(
        self,
        trace: ReplayTrace,
        objective: OracleObjective = OracleObjective.MAXIMIZE_UTILITY,
        constraints: OracleConstraints | None = None,
    ) -> OraclePlan:
        constraints = constraints or OracleConstraints()
        best_stop_index = self._select_stop_index(trace, objective, constraints)
        decisions: list[OracleDecision] = []
        cumulative_latency = 0.0
        cumulative_compute = 0.0
        peak_memory = 0.0

        for index, layer in enumerate(trace.layers):
            cumulative_latency += max(0.0, layer.latency_ms)
            cumulative_compute += estimate_compute_units(layer)
            memory = layer_memory_mb(layer)
            if memory is not None:
                peak_memory = max(peak_memory, memory)
            action = (
                SchedulerAction.EARLY_EXIT
                if best_stop_index is not None and index >= best_stop_index
                else SchedulerAction.CONTINUE
            )
            decisions.append(
                OracleDecision(
                    index=index,
                    layer_name=layer.layer_name,
                    action=str(action),
                    utility=_prefix_utility(
                        trace=trace,
                        stop_index=index,
                        objective=objective,
                        constraints=constraints,
                    ),
                    cumulative_latency_ms=cumulative_latency,
                    cumulative_compute_units=cumulative_compute,
                    peak_memory_mb=peak_memory,
                )
            )

        return OraclePlan(
            objective=objective,
            decisions=tuple(decisions),
            utility=_prefix_utility(trace, best_stop_index, objective, constraints),
            stop_index=best_stop_index,
        )

    def _select_stop_index(
        self,
        trace: ReplayTrace,
        objective: OracleObjective,
        constraints: OracleConstraints,
    ) -> int | None:
        if not trace.layers:
            return None

        candidates = list(range(len(trace.layers))) + [None]
        feasible = [
            index
            for index in candidates
            if _satisfies_constraints(trace=trace, stop_index=index, constraints=constraints)
        ]
        if not feasible:
            feasible = candidates
        return max(
            feasible,
            key=lambda index: _prefix_utility(
                trace=trace,
                stop_index=index,
                objective=objective,
                constraints=constraints,
            ),
        )


def _prefix_utility(
    trace: ReplayTrace,
    stop_index: int | None,
    objective: OracleObjective,
    constraints: OracleConstraints,
) -> float:
    latency, compute, memory = _prefix_costs(trace, stop_index)
    quality = _quality_at_prefix(trace, stop_index)
    constraint_penalty = _constraint_penalty(latency, compute, memory, constraints)

    if objective == OracleObjective.MAXIMIZE_QUALITY:
        return quality - constraint_penalty
    if objective == OracleObjective.MINIMIZE_LATENCY:
        return -latency - constraint_penalty
    if objective == OracleObjective.MINIMIZE_FLOPS:
        return -compute - constraint_penalty
    if objective == OracleObjective.MINIMIZE_MEMORY:
        return -memory - constraint_penalty
    if objective == OracleObjective.BALANCED:
        return quality - 0.001 * latency - 0.001 * compute - 0.0001 * memory - constraint_penalty
    return quality - 0.001 * latency - 0.001 * compute - 0.0001 * memory - constraint_penalty


def _prefix_costs(trace: ReplayTrace, stop_index: int | None) -> tuple[float, float, float]:
    layers = trace.layers if stop_index is None else trace.layers[: stop_index + 1]
    latency = sum(max(0.0, layer.latency_ms) for layer in layers)
    compute = sum(estimate_compute_units(layer) for layer in layers)
    memory = max((layer_memory_mb(layer) or 0.0 for layer in layers), default=0.0)
    return latency, compute, memory


def _quality_at_prefix(trace: ReplayTrace, stop_index: int | None) -> float:
    if stop_index is None or not trace.layers:
        return max(0.0, trace.final_utility + 0.001 * trace.total_latency_ms)
    fraction = (stop_index + 1) / len(trace.layers)
    return max(0.0, trace.final_utility + 0.001 * trace.total_latency_ms) * fraction


def _satisfies_constraints(
    trace: ReplayTrace,
    stop_index: int | None,
    constraints: OracleConstraints,
) -> bool:
    latency, compute, memory = _prefix_costs(trace, stop_index)
    if constraints.latency_budget_ms is not None and latency > constraints.latency_budget_ms:
        return False
    if constraints.compute_budget_units is not None and compute > constraints.compute_budget_units:
        return False
    if constraints.memory_budget_mb is not None and memory > constraints.memory_budget_mb:
        return False
    return True


def _constraint_penalty(
    latency: float,
    compute: float,
    memory: float,
    constraints: OracleConstraints,
) -> float:
    penalty = 0.0
    if constraints.latency_budget_ms is not None:
        penalty += max(0.0, latency - constraints.latency_budget_ms)
    if constraints.compute_budget_units is not None:
        penalty += max(0.0, compute - constraints.compute_budget_units)
    if constraints.memory_budget_mb is not None:
        penalty += max(0.0, memory - constraints.memory_budget_mb)
    return penalty
