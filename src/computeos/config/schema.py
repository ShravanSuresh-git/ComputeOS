"""Typed configuration objects used by Hydra and direct Python callers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Hugging Face model and tokenizer settings."""

    name: str = "distilgpt2"
    revision: str | None = None
    torch_dtype: str = "auto"
    device_map: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler selection and policy parameters."""

    name: str = "heuristic"
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetryConfig:
    """Telemetry collection and logging settings."""

    enabled: bool = True
    capture_activations: bool = True
    capture_attention_entropy: bool = True
    capture_memory: bool = True
    log_to_wandb: bool = False
    wandb_project: str = "computeos"
    wandb_entity: str | None = None
    sample_rate: float = 1.0
    export_path: str | None = None
    export_format: str | None = None
    pretty_report: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime controls for inference execution."""

    max_new_tokens: int = 32
    use_cache: bool = True
    seed: int = 17
    autocast: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark runner settings."""

    name: str = "prompt_smoke"
    prompts: list[str] = field(default_factory=lambda: ["ComputeOS is"])
    batch_size: int = 1
    limit: int | None = None
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeOSConfig:
    """Top-level ComputeOS experiment config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
