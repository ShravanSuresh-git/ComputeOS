"""Serialization helpers for telemetry records."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from computeos.telemetry.metrics import LayerTelemetry, ModelTelemetry


def model_telemetry_to_dict(telemetry: ModelTelemetry) -> dict[str, Any]:
    """Convert telemetry into a JSON-serializable dictionary."""

    payload = asdict(telemetry)
    payload["scheduler_decisions"] = [
        {
            **decision,
            "action": str(decision["action"]),
        }
        for decision in payload["scheduler_decisions"]
    ]
    return payload


def layer_telemetry_to_row(
    telemetry: ModelTelemetry,
    layer: LayerTelemetry,
    run_index: int,
    layer_index: int,
) -> dict[str, object]:
    """Flatten one layer event into a CSV-friendly row."""

    stats = layer.activation_stats
    return {
        "run_index": run_index,
        "model_name": telemetry.model_name,
        "layer_index": layer_index,
        "layer_name": layer.layer_name,
        "layer_type": layer.layer_type,
        "latency_ms": layer.latency_ms,
        "attention_entropy": layer.attention_entropy,
        "memory_allocated_bytes": layer.memory_allocated_bytes,
        "memory_reserved_bytes": layer.memory_reserved_bytes,
        "process_rss_bytes": layer.process_rss_bytes,
        "activation_mean": None if stats is None else stats.mean,
        "activation_std": None if stats is None else stats.std,
        "activation_min": None if stats is None else stats.min,
        "activation_max": None if stats is None else stats.max,
        "activation_l2_norm": None if stats is None else stats.l2_norm,
        "activation_numel": None if stats is None else stats.numel,
        "total_latency_ms": telemetry.total_latency_ms,
        "peak_memory_bytes": telemetry.peak_memory_bytes,
        "peak_process_rss_bytes": telemetry.peak_process_rss_bytes,
        "scheduler_decision_count": len(telemetry.scheduler_decisions),
        "token_confidence_mean": _mean_or_none(telemetry.confidence_scores),
        "scheduler_confidence_mean": _mean_or_none(telemetry.scheduler_confidence_scores),
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
