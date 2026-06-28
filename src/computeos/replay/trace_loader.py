"""Trace loading for Counterfactual Runtime Intelligence.

The first CRI implementation treats existing `ModelTelemetry` as the canonical
completed trace format. This preserves backwards compatibility while creating a
stable replay model that future runtime event streams can populate directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.telemetry.metrics import LayerTelemetry, ModelTelemetry


class RuntimeEventType(StrEnum):
    """Replay event kinds derived from completed telemetry."""

    REQUEST_STARTED = "request_started"
    LAYER_FINISHED = "layer_finished"
    SCHEDULER_DECISION = "scheduler_decision"
    REQUEST_FINISHED = "request_finished"


@dataclass(frozen=True)
class RuntimeEvent:
    """A deterministic replay event."""

    index: int
    event_type: RuntimeEventType
    timestamp_offset_ms: float
    layer_name: str | None = None
    layer_type: str | None = None
    latency_ms: float | None = None
    action: str | None = None
    utility: float | None = None
    memory_mb: float | None = None
    compute_units: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayTrace:
    """Completed inference trace used for replay and counterfactual analysis."""

    model_name: str
    events: tuple[RuntimeEvent, ...]
    layers: tuple[LayerTelemetry, ...]
    decisions: tuple[SchedulerDecision, ...]
    total_latency_ms: float
    peak_memory_mb: float
    total_compute_units: float
    final_utility: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def decision_events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.event_type == RuntimeEventType.SCHEDULER_DECISION
        )

    def layer_events(self) -> tuple[RuntimeEvent, ...]:
        return tuple(
            event for event in self.events if event.event_type == RuntimeEventType.LAYER_FINISHED
        )


class TraceLoader:
    """Load replay traces from existing ComputeOS telemetry."""

    def from_telemetry(self, telemetry: ModelTelemetry) -> ReplayTrace:
        events: list[RuntimeEvent] = [
            RuntimeEvent(
                index=0,
                event_type=RuntimeEventType.REQUEST_STARTED,
                timestamp_offset_ms=0.0,
                metadata={"model_name": telemetry.model_name, **telemetry.metadata},
            )
        ]
        elapsed_ms = 0.0
        event_index = 1
        total_compute_units = 0.0
        peak_memory_mb = _peak_memory_mb(telemetry)

        for layer_index, layer in enumerate(telemetry.layers):
            elapsed_ms += max(0.0, layer.latency_ms)
            compute_units = estimate_compute_units(layer)
            total_compute_units += compute_units
            events.append(
                RuntimeEvent(
                    index=event_index,
                    event_type=RuntimeEventType.LAYER_FINISHED,
                    timestamp_offset_ms=elapsed_ms,
                    layer_name=layer.layer_name,
                    layer_type=layer.layer_type,
                    latency_ms=layer.latency_ms,
                    memory_mb=layer_memory_mb(layer),
                    compute_units=compute_units,
                    metadata={
                        "layer_index": layer_index,
                        "attention_entropy": layer.attention_entropy,
                        "activation_stats": layer.activation_stats,
                        **layer.metadata,
                    },
                )
            )
            event_index += 1

            if layer_index < len(telemetry.scheduler_decisions):
                decision = telemetry.scheduler_decisions[layer_index]
                utility = decision_utility(decision, layer)
                events.append(
                    RuntimeEvent(
                        index=event_index,
                        event_type=RuntimeEventType.SCHEDULER_DECISION,
                        timestamp_offset_ms=elapsed_ms,
                        layer_name=decision.layer_name or layer.layer_name,
                        action=str(decision.action),
                        utility=utility,
                        memory_mb=layer_memory_mb(layer),
                        compute_units=compute_units,
                        metadata={
                            "decision_index": layer_index,
                            "reason": decision.reason,
                            "confidence": decision.confidence,
                            **decision.metadata,
                        },
                    )
                )
                event_index += 1

        total_latency_ms = telemetry.total_latency_ms
        if total_latency_ms is None:
            total_latency_ms = sum(max(0.0, layer.latency_ms) for layer in telemetry.layers)
        final_utility = trace_utility(
            latency_ms=total_latency_ms,
            memory_mb=peak_memory_mb,
            compute_units=total_compute_units,
            quality_proxy=quality_proxy(telemetry),
        )
        events.append(
            RuntimeEvent(
                index=event_index,
                event_type=RuntimeEventType.REQUEST_FINISHED,
                timestamp_offset_ms=total_latency_ms,
                utility=final_utility,
                memory_mb=peak_memory_mb,
                compute_units=total_compute_units,
                metadata={"quality_proxy": quality_proxy(telemetry)},
            )
        )
        return ReplayTrace(
            model_name=telemetry.model_name,
            events=tuple(events),
            layers=tuple(telemetry.layers),
            decisions=tuple(telemetry.scheduler_decisions),
            total_latency_ms=total_latency_ms,
            peak_memory_mb=peak_memory_mb,
            total_compute_units=total_compute_units,
            final_utility=final_utility,
            metadata=dict(telemetry.metadata),
        )


def estimate_compute_units(layer: LayerTelemetry) -> float:
    """Estimate compute from telemetry when true FLOPs are unavailable."""

    if layer.activation_stats is None:
        return 1.0
    return max(1.0, layer.activation_stats.numel / 1024.0)


def layer_memory_mb(layer: LayerTelemetry) -> float | None:
    value = layer.memory_allocated_bytes or layer.process_rss_bytes
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def quality_proxy(telemetry: ModelTelemetry) -> float:
    """Estimate trace quality from available confidence and decision metadata.

    This is not a replacement for benchmark accuracy. CRI uses it only for
    counterfactual ranking when no task score is available.
    """

    if telemetry.confidence_scores:
        return sum(telemetry.confidence_scores) / len(telemetry.confidence_scores)
    pvs_values = [
        float(decision.metadata["prediction"]["expected_improvement"])
        for decision in telemetry.scheduler_decisions
        if isinstance(decision.metadata.get("prediction"), dict)
        and "expected_improvement" in decision.metadata["prediction"]
    ]
    if pvs_values:
        return sum(pvs_values) / len(pvs_values)
    return 0.5


def decision_utility(decision: SchedulerDecision, layer: LayerTelemetry) -> float:
    prediction = decision.metadata.get("prediction")
    if isinstance(prediction, dict) and "expected_net_value" in prediction:
        return float(prediction["expected_net_value"])
    confidence = decision.confidence if decision.confidence is not None else 0.5
    cost = 0.001 * layer.latency_ms + 0.01 * estimate_compute_units(layer)
    return confidence - cost


def trace_utility(
    latency_ms: float,
    memory_mb: float,
    compute_units: float,
    quality_proxy: float,
) -> float:
    """Default scalar utility used for offline comparisons."""

    return quality_proxy - 0.001 * latency_ms - 0.0001 * memory_mb - 0.001 * compute_units


def first_stopping_index(decisions: tuple[SchedulerDecision, ...]) -> int | None:
    for index, decision in enumerate(decisions):
        if decision.action == SchedulerAction.EARLY_EXIT:
            return index
    return None


def _peak_memory_mb(telemetry: ModelTelemetry) -> float:
    if telemetry.peak_process_rss_bytes is not None:
        return telemetry.peak_process_rss_bytes / (1024.0 * 1024.0)
    if telemetry.peak_memory_bytes is not None:
        return telemetry.peak_memory_bytes / (1024.0 * 1024.0)
    values = [layer_memory_mb(layer) for layer in telemetry.layers]
    return max((value for value in values if value is not None), default=0.0)
