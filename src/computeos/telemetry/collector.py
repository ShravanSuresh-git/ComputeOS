"""Central telemetry collector used by hooks and execution engines."""

from __future__ import annotations

from computeos.scheduling.decision import SchedulerDecision
from computeos.telemetry.metrics import LayerTelemetry, ModelTelemetry


class TelemetryCollector:
    """Mutable collection boundary for one inference request."""

    def __init__(self, model_name: str, request_id: str | None = None) -> None:
        import uuid

        self.model_telemetry = ModelTelemetry(
            model_name=model_name,
            request_id=request_id or str(uuid.uuid4()),
        )

    def record_layer(self, telemetry: LayerTelemetry) -> None:
        self.model_telemetry.layers.append(telemetry)

    def record_decision(self, decision: SchedulerDecision) -> None:
        self.model_telemetry.scheduler_decisions.append(decision)
        if decision.confidence is not None:
            self.model_telemetry.scheduler_confidence_scores.append(decision.confidence)

    def record_confidence_scores(self, scores: list[float]) -> None:
        if self.model_telemetry.confidence_scores:
            return
        self.model_telemetry.confidence_scores.extend(scores)

    def push_confidence_score(self, score: float) -> None:
        """Append a single per-token confidence score during generation."""

        self.model_telemetry.confidence_scores.append(score)

    def finish(
        self,
        total_latency_ms: float,
        peak_memory_bytes: int | None = None,
        peak_process_rss_bytes: int | None = None,
    ) -> ModelTelemetry:
        self.model_telemetry.total_latency_ms = total_latency_ms
        self.model_telemetry.peak_memory_bytes = peak_memory_bytes
        self.model_telemetry.peak_process_rss_bytes = peak_process_rss_bytes
        return self.model_telemetry
