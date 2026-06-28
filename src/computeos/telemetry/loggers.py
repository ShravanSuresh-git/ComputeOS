"""Telemetry logging backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from computeos.telemetry.metrics import ModelTelemetry
from computeos.telemetry.serialization import layer_telemetry_to_row, model_telemetry_to_dict


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


class CompositeTelemetryLogger(TelemetryLogger):
    """Fan-out logger for experiments that need multiple sinks."""

    def __init__(self, loggers: list[TelemetryLogger]) -> None:
        self._loggers = loggers

    def log(self, telemetry: ModelTelemetry) -> None:
        for logger in self._loggers:
            logger.log(telemetry)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()


class JsonTelemetryLogger(TelemetryLogger):
    """Write model telemetry records as a JSON array or JSONL stream."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._records: list[dict[str, Any]] = []

    def log(self, telemetry: ModelTelemetry) -> None:
        self._records.append(model_telemetry_to_dict(telemetry))

    def close(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.suffix == ".jsonl":
            with self._path.open("w", encoding="utf-8") as file:
                for record in self._records:
                    file.write(json.dumps(record) + "\n")
            return
        with self._path.open("w", encoding="utf-8") as file:
            json.dump(self._records, file, indent=2)


class CsvTelemetryLogger(TelemetryLogger):
    """Write one CSV row per layer event."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._rows: list[dict[str, object]] = []
        self._run_index = 0

    def log(self, telemetry: ModelTelemetry) -> None:
        for layer_index, layer in enumerate(telemetry.layers):
            self._rows.append(
                layer_telemetry_to_row(
                    telemetry=telemetry,
                    layer=layer,
                    run_index=self._run_index,
                    layer_index=layer_index,
                )
            )
        self._run_index += 1

    def close(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(_CSV_FIELDNAMES)
        with self._path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)


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


_CSV_FIELDNAMES = (
    "run_index",
    "model_name",
    "layer_index",
    "layer_name",
    "layer_type",
    "latency_ms",
    "attention_entropy",
    "memory_allocated_bytes",
    "memory_reserved_bytes",
    "process_rss_bytes",
    "activation_mean",
    "activation_std",
    "activation_min",
    "activation_max",
    "activation_l2_norm",
    "activation_numel",
    "total_latency_ms",
    "peak_memory_bytes",
    "peak_process_rss_bytes",
    "scheduler_decision_count",
    "token_confidence_mean",
    "scheduler_confidence_mean",
)
