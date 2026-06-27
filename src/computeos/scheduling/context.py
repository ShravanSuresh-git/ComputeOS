"""Context passed from instrumentation to schedulers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from computeos.telemetry.metrics import LayerTelemetry, ModelTelemetry


@dataclass(frozen=True)
class SchedulerContext:
    """Immutable view of the current inference state."""

    step_index: int
    layer_name: str | None
    layer_telemetry: LayerTelemetry | None
    model_telemetry: ModelTelemetry
    model_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
