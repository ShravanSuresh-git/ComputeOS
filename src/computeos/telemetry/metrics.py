"""Typed telemetry records."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from time import time

from computeos.scheduling.decision import SchedulerDecision


@dataclass(frozen=True)
class ActivationStats:
    """Summary statistics for a tensor activation."""

    mean: float
    std: float
    min: float
    max: float
    l2_norm: float
    numel: int


@dataclass
class LayerTelemetry:
    """Telemetry emitted for one module invocation."""

    layer_name: str
    layer_type: str
    latency_ms: float
    activation_stats: ActivationStats | None = None
    attention_entropy: float | None = None
    attention_entropy_available: bool = False
    memory_allocated_bytes: int | None = None
    memory_reserved_bytes: int | None = None
    process_rss_bytes: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ModelTelemetry:
    """Telemetry for one end-to-end inference call."""

    model_name: str
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time)
    layers: list[LayerTelemetry] = field(default_factory=list)
    scheduler_decisions: list[SchedulerDecision] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    scheduler_confidence_scores: list[float] = field(default_factory=list)
    total_latency_ms: float | None = None
    peak_memory_bytes: int | None = None
    peak_process_rss_bytes: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
