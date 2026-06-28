"""Counterfactual scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ScenarioType(StrEnum):
    """Supported counterfactual experiment types."""

    CONTINUE_EXTRA_LAYERS = "continue_extra_layers"
    INCREASE_REASONING_BUDGET = "increase_reasoning_budget"
    DECREASE_REASONING_BUDGET = "decrease_reasoning_budget"
    CHANGE_LATENCY_BUDGET = "change_latency_budget"
    CHANGE_COMPUTE_BUDGET = "change_compute_budget"
    CHANGE_MEMORY_BUDGET = "change_memory_budget"
    REPLACE_SCHEDULER = "replace_scheduler"
    CHANGE_STOPPING_CRITERION = "change_stopping_criterion"


@dataclass(frozen=True)
class CounterfactualScenario:
    """A deterministic CRI scenario."""

    name: str
    scenario_type: ScenarioType
    extra_layers: int = 0
    latency_budget_ms: float | None = None
    compute_budget_units: float | None = None
    memory_budget_mb: float | None = None
    scheduler_name: str | None = None
    stopping_threshold: float | None = None
    parameters: dict[str, object] = field(default_factory=dict)
