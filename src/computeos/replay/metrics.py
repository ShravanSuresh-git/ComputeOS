"""Research metrics for Counterfactual Runtime Intelligence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayMetrics:
    """Scalar metrics used for replay and counterfactual comparisons."""

    utility: float
    latency_ms: float
    compute_units: float
    memory_mb: float
    quality_proxy: float
    budget_efficiency: float
    expected_information_gain: float
    utility_per_flop: float
    utility_per_joule: float
    counterfactual_improvement: float
    decision_stability: float
    stopping_accuracy: float
    oracle_gap: float


def budget_efficiency(utility: float, compute_units: float, latency_ms: float) -> float:
    denominator = max(1e-9, compute_units + latency_ms)
    return utility / denominator


def expected_information_gain(quality_after: float, quality_before: float) -> float:
    return quality_after - quality_before


def utility_per_flop(utility: float, compute_units: float) -> float:
    return utility / max(1e-9, compute_units)


def utility_per_joule(utility: float, joules: float | None = None) -> float:
    estimated_joules = 1.0 if joules is None else max(1e-9, joules)
    return utility / estimated_joules


def decision_stability(actions_a: list[str], actions_b: list[str]) -> float:
    if not actions_a and not actions_b:
        return 1.0
    length = max(len(actions_a), len(actions_b))
    matches = 0
    for index in range(length):
        left = actions_a[index] if index < len(actions_a) else None
        right = actions_b[index] if index < len(actions_b) else None
        if left == right:
            matches += 1
    return matches / length
