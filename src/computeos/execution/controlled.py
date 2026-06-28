"""Controlled runtime for action-applying adaptive inference research.

The Hugging Face `generate` path is intentionally compatibility-first and
observational. This module provides a small controlled runtime for models whose
layers can be executed explicitly, allowing ComputeOS to enforce scheduler
actions in tests, demos, and future model adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn

from computeos.instrumentation.layers import discover_transformer_layers
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.telemetry.collector import TelemetryCollector
from computeos.telemetry.metrics import LayerTelemetry, ModelTelemetry
from computeos.telemetry.stats import activation_stats, attention_entropy


@dataclass(frozen=True)
class RuntimeBudget:
    """Hard resource budget for a controlled runtime call."""

    max_layers: int | None = None
    max_latency_ms: float | None = None
    max_compute_units: float | None = None


@dataclass(frozen=True)
class ActionResult:
    """Result of applying or rejecting a scheduler action."""

    requested_action: SchedulerAction
    applied: bool
    reason: str
    layer_name: str | None = None


@dataclass(frozen=True)
class ControlledExecutionResult:
    """Output and telemetry from controlled layer execution."""

    output: Any
    telemetry: ModelTelemetry
    action_results: tuple[ActionResult, ...]


class ControlledForwardRuntime:
    """Execute model layers explicitly and enforce scheduler decisions.

    Supported actions:

    - `SKIP_LAYER` at the pre-layer decision point
    - `EARLY_EXIT` at either pre-layer or post-layer decision points
    - hard layer, latency, and compute budgets
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: Scheduler,
        model_name: str = "controlled",
        layers: list[tuple[str, nn.Module]] | None = None,
        budget: RuntimeBudget | None = None,
    ) -> None:
        self._model = model
        self._scheduler = scheduler
        self._model_name = model_name
        self._layers = layers
        self._budget = budget or RuntimeBudget()

    @torch.inference_mode()
    def run(self, inputs: Any) -> ControlledExecutionResult:
        """Run controlled forward execution."""

        self._scheduler.reset()
        collector = TelemetryCollector(model_name=self._model_name)
        action_results: list[ActionResult] = []
        layers = self._layers or _discover_or_children(self._model)
        output = inputs
        started_at = perf_counter()
        compute_units = 0.0

        for index, (name, module) in enumerate(layers):
            budget_decision = _budget_decision(
                budget=self._budget,
                layer_index=index,
                elapsed_ms=(perf_counter() - started_at) * 1000.0,
                compute_units=compute_units,
                layer_name=name,
            )
            if budget_decision is not None:
                collector.record_decision(budget_decision)
                action_results.append(
                    ActionResult(
                        requested_action=budget_decision.action,
                        applied=True,
                        reason=budget_decision.reason,
                        layer_name=name,
                    )
                )
                break

            pre_decision = self._scheduler.decide(
                SchedulerContext(
                    step_index=len(collector.model_telemetry.scheduler_decisions),
                    layer_name=name,
                    layer_telemetry=None,
                    model_telemetry=collector.model_telemetry,
                    metadata={"decision_point": "pre_layer"},
                )
            )
            if pre_decision.action == SchedulerAction.SKIP_LAYER:
                collector.record_decision(
                    _with_action_result(pre_decision, applied=True, reason="layer skipped")
                )
                action_results.append(
                    ActionResult(
                        requested_action=pre_decision.action,
                        applied=True,
                        reason="layer skipped",
                        layer_name=name,
                    )
                )
                continue
            if pre_decision.action == SchedulerAction.EARLY_EXIT:
                collector.record_decision(
                    _with_action_result(
                        pre_decision,
                        applied=True,
                        reason="early exit before layer",
                    )
                )
                action_results.append(
                    ActionResult(
                        requested_action=pre_decision.action,
                        applied=True,
                        reason="early exit before layer",
                        layer_name=name,
                    )
                )
                break
            if pre_decision.action not in {SchedulerAction.RECORD_ONLY, SchedulerAction.CONTINUE}:
                collector.record_decision(
                    _with_action_result(
                        pre_decision,
                        applied=False,
                        reason="unsupported pre-layer action",
                    )
                )

            layer_started = perf_counter()
            output = module(output)
            latency_ms = (perf_counter() - layer_started) * 1000.0
            layer_telemetry = LayerTelemetry(
                layer_name=name,
                layer_type=module.__class__.__name__,
                latency_ms=latency_ms,
                activation_stats=activation_stats(output),
                attention_entropy=attention_entropy(output),
            )
            compute_units += _compute_units(layer_telemetry)
            collector.record_layer(layer_telemetry)

            post_decision = self._scheduler.decide(
                SchedulerContext(
                    step_index=len(collector.model_telemetry.scheduler_decisions),
                    layer_name=name,
                    layer_telemetry=layer_telemetry,
                    model_telemetry=collector.model_telemetry,
                    metadata={"decision_point": "post_layer"},
                )
            )
            if post_decision.action == SchedulerAction.EARLY_EXIT:
                collector.record_decision(
                    _with_action_result(
                        post_decision,
                        applied=True,
                        reason="early exit after layer",
                    )
                )
                action_results.append(
                    ActionResult(
                        requested_action=post_decision.action,
                        applied=True,
                        reason="early exit after layer",
                        layer_name=name,
                    )
                )
                break
            if post_decision.action == SchedulerAction.SKIP_LAYER:
                collector.record_decision(
                    _with_action_result(
                        post_decision,
                        applied=False,
                        reason="skip requested after layer execution",
                    )
                )
            else:
                collector.record_decision(
                    _with_action_result(post_decision, applied=False, reason="recorded")
                )
            self._scheduler.observe(
                SchedulerContext(
                    step_index=len(collector.model_telemetry.scheduler_decisions),
                    layer_name=name,
                    layer_telemetry=layer_telemetry,
                    model_telemetry=collector.model_telemetry,
                ),
                post_decision,
            )

        telemetry = collector.finish(total_latency_ms=(perf_counter() - started_at) * 1000.0)
        telemetry.metadata["runtime"] = "controlled_forward"
        telemetry.metadata["compute_units"] = compute_units
        return ControlledExecutionResult(
            output=output,
            telemetry=telemetry,
            action_results=tuple(action_results),
        )


def _discover_or_children(model: nn.Module) -> list[tuple[str, nn.Module]]:
    layers = discover_transformer_layers(model)
    if layers:
        return layers
    return [(name, module) for name, module in model.named_children()]


def _budget_decision(
    budget: RuntimeBudget,
    layer_index: int,
    elapsed_ms: float,
    compute_units: float,
    layer_name: str,
) -> SchedulerDecision | None:
    if budget.max_layers is not None and layer_index >= budget.max_layers:
        return SchedulerDecision(
            action=SchedulerAction.EARLY_EXIT,
            layer_name=layer_name,
            reason="max layer budget reached",
            metadata={"action_result": {"applied": True, "reason": "budget enforcement"}},
        )
    if budget.max_latency_ms is not None and elapsed_ms >= budget.max_latency_ms:
        return SchedulerDecision(
            action=SchedulerAction.EARLY_EXIT,
            layer_name=layer_name,
            reason="latency budget reached",
            metadata={"action_result": {"applied": True, "reason": "budget enforcement"}},
        )
    if budget.max_compute_units is not None and compute_units >= budget.max_compute_units:
        return SchedulerDecision(
            action=SchedulerAction.EARLY_EXIT,
            layer_name=layer_name,
            reason="compute budget reached",
            metadata={"action_result": {"applied": True, "reason": "budget enforcement"}},
        )
    return None


def _with_action_result(
    decision: SchedulerDecision,
    applied: bool,
    reason: str,
) -> SchedulerDecision:
    metadata = dict(decision.metadata)
    metadata["action_result"] = {"applied": applied, "reason": reason}
    return SchedulerDecision(
        action=decision.action,
        layer_name=decision.layer_name,
        confidence=decision.confidence,
        reason=decision.reason,
        metadata=metadata,
    )


def _compute_units(layer: LayerTelemetry) -> float:
    if layer.activation_stats is None:
        return 1.0
    return max(1.0, layer.activation_stats.numel / 1024.0)
