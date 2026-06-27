"""Hydra experiment runner."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from computeos.benchmarks.registry import default_benchmark_registry
from computeos.config.schema import (
    BenchmarkConfig,
    ComputeOSConfig,
    ExecutionConfig,
    ModelConfig,
    SchedulerConfig,
    TelemetryConfig,
)
from computeos.execution.engine import InferenceEngine
from computeos.execution.model_loader import load_hf_causal_lm
from computeos.scheduling.registry import default_scheduler_registry
from computeos.telemetry.loggers import (
    InMemoryTelemetryLogger,
    TelemetryLogger,
    WandbTelemetryLogger,
)


def run_experiment(config: ComputeOSConfig) -> list[dict[str, Any]]:
    """Run a ComputeOS benchmark from a typed config object."""

    loaded = load_hf_causal_lm(config.model)
    scheduler = default_scheduler_registry().create(config.scheduler)
    engine = InferenceEngine(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        model_name=loaded.name,
        scheduler=scheduler,
        execution_config=config.execution,
        telemetry_config=config.telemetry,
    )
    benchmark = default_benchmark_registry().create(config.benchmark)
    logger = _build_logger(config)
    try:
        results = benchmark.run(engine)
        for result in results:
            logger.log(result.execution.telemetry)
        return [
            {
                "prompt": result.item.prompt,
                "generated_text": result.execution.generated_text,
                "score": result.score,
                "telemetry": result.execution.telemetry,
            }
            for result in results
        ]
    finally:
        logger.close()


def _build_logger(config: ComputeOSConfig) -> TelemetryLogger:
    if config.telemetry.log_to_wandb:
        return WandbTelemetryLogger(
            project=config.telemetry.wandb_project,
            entity=config.telemetry.wandb_entity,
            config=asdict(config),
        )
    return InMemoryTelemetryLogger()


def _from_omegaconf(cfg: Any) -> ComputeOSConfig:
    return ComputeOSConfig(
        model=ModelConfig(**dict(cfg.model)),
        scheduler=SchedulerConfig(
            name=str(cfg.scheduler.name),
            parameters=dict(cfg.scheduler.get("parameters", {})),
        ),
        telemetry=TelemetryConfig(**dict(cfg.telemetry)),
        execution=ExecutionConfig(**dict(cfg.execution)),
        benchmark=BenchmarkConfig(**dict(cfg.benchmark)),
    )


def main() -> None:
    """CLI entrypoint. Imports Hydra lazily so tests need not install it."""

    import hydra
    from omegaconf import DictConfig

    config_path = str(Path(__file__).resolve().parents[3] / "conf")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="config")
    def _hydra_main(cfg: DictConfig) -> None:
        results = run_experiment(_from_omegaconf(cfg))
        for result in results:
            print(result["generated_text"])

    _hydra_main()


if __name__ == "__main__":
    main()
