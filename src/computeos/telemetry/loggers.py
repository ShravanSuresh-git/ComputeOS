"""Telemetry logging backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from computeos.telemetry.metrics import ModelTelemetry


class TelemetryLogger(ABC):
    """Sink interface for telemetry backends."""

    @abstractmethod
    def log(self, telemetry: ModelTelemetry) -> None:
        """Persist or emit telemetry for one inference call."""

    def close(self) -> None:
        """Release logger resources."""


class InMemoryTelemetryLogger(TelemetryLogger):
    """Test and notebook-friendly telemetry sink."""

    def __init__(self) -> None:
        self.records: list[ModelTelemetry] = []

    def log(self, telemetry: ModelTelemetry) -> None:
        self.records.append(telemetry)


class WandbTelemetryLogger(TelemetryLogger):
    """Weights & Biases telemetry logger with lazy import."""

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, entity=entity, config=config)

    def log(self, telemetry: ModelTelemetry) -> None:
        payload: dict[str, Any] = {
            "latency/total_ms": telemetry.total_latency_ms,
            "memory/peak_bytes": telemetry.peak_memory_bytes,
            "memory/peak_process_rss_bytes": telemetry.peak_process_rss_bytes,
            "scheduler/decision_count": len(telemetry.scheduler_decisions),
            "layers/count": len(telemetry.layers),
        }
        if telemetry.confidence_scores:
            payload["confidence/token_mean"] = sum(telemetry.confidence_scores) / len(
                telemetry.confidence_scores
            )
        if telemetry.scheduler_confidence_scores:
            payload["confidence/scheduler_mean"] = sum(
                telemetry.scheduler_confidence_scores
            ) / len(telemetry.scheduler_confidence_scores)
        for index, layer in enumerate(telemetry.layers):
            prefix = f"layers/{index}/{layer.layer_name}"
            payload[f"{prefix}/latency_ms"] = layer.latency_ms
            payload[f"{prefix}/attention_entropy"] = layer.attention_entropy
            payload[f"{prefix}/memory_process_rss_bytes"] = layer.process_rss_bytes
            if layer.activation_stats is not None:
                for key, value in asdict(layer.activation_stats).items():
                    payload[f"{prefix}/activation/{key}"] = value
        self._wandb.log(payload)

    def close(self) -> None:
        self._run.finish()
