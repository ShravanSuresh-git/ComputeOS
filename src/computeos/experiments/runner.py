"""Hydra experiment runner."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

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
    CompositeTelemetryLogger,
    CsvTelemetryLogger,
    InMemoryTelemetryLogger,
    JsonTelemetryLogger,
    TelemetryLogger,
    WandbTelemetryLogger,
)
from computeos.telemetry.reports import print_telemetry_report


def run_experiment(config: ComputeOSConfig) -> list[dict[str, Any]]:
    """Run a ComputeOS benchmark from a typed config object."""

    loaded = load_hf_causal_lm(config.model)
    scheduler = default_scheduler_registry().create(config.scheduler)
    if config.benchmark.batch_size > 1:
        import warnings

        warnings.warn(
            f"BenchmarkConfig.batch_size={config.benchmark.batch_size} is set "
            "but the engine processes one prompt at a time. "
            "Results are sequential. Batching support is tracked in TECHNICAL_DEBT.md.",
            UserWarning,
            stacklevel=2,
        )
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
    results = []
    try:
        results = benchmark.run(engine)
        for result in results:
            logger.log(result.execution.telemetry)
    finally:
        logger.close()
    payload = [
        {
            "prompt": result.item.prompt,
            "generated_text": result.execution.generated_text,
            "score": result.score,
            "telemetry": result.execution.telemetry,
        }
        for result in results
    ]
    _snapshot_artifacts(config, payload)
    return payload


def _snapshot_artifacts(config: ComputeOSConfig, results: list[dict[str, Any]]) -> None:
    try:
        from computeos.experiments.artifacts import ArtifactStore
        from computeos.experiments.comparison import ComparisonReport

        store = ArtifactStore(output_dir=Path("outputs"))
        store.snapshot_config(asdict(config))
        store.snapshot_env()
        for result in results:
            telemetry = result.get("telemetry")
            if telemetry is not None:
                store.snapshot_telemetry(telemetry)
        comparison_report = ComparisonReport(
            rows=[
                {
                    "scheduler": str(config.scheduler.name),
                    "prompt": str(result["prompt"]),
                    "generated_text": str(result["generated_text"]),
                }
                for result in results
                if isinstance(result, dict)
            ]
        )
        store.snapshot_report(comparison_report)
    except Exception:
        return


def _build_logger(config: ComputeOSConfig) -> TelemetryLogger:
    loggers: list[TelemetryLogger] = []
    if config.telemetry.log_to_wandb:
        loggers.append(
            WandbTelemetryLogger(
                project=config.telemetry.wandb_project,
                entity=config.telemetry.wandb_entity,
                config=asdict(config),
            )
        )
    if config.telemetry.export_path is not None:
        loggers.append(_build_export_logger(config.telemetry))
    if not loggers:
        return InMemoryTelemetryLogger()
    if len(loggers) == 1:
        return loggers[0]
    return CompositeTelemetryLogger(loggers)


def _build_export_logger(config: TelemetryConfig) -> TelemetryLogger:
    if config.export_path is None:
        raise ValueError("export_path is required when building an export logger.")
    export_format = (config.export_format or Path(config.export_path).suffix.lstrip(".")).lower()
    if export_format in {"json", "jsonl"}:
        return JsonTelemetryLogger(config.export_path)
    if export_format == "csv":
        return CsvTelemetryLogger(config.export_path)
    raise ValueError(f"Unsupported telemetry export format: {export_format}")


def _from_omegaconf(cfg: object) -> ComputeOSConfig:
    cfg_any = cast(Any, cfg)
    return ComputeOSConfig(
        model=ModelConfig(**dict(cfg_any.model)),
        scheduler=SchedulerConfig(
            name=str(cfg_any.scheduler.name),
            parameters=dict(cfg_any.scheduler.get("parameters", {})),
        ),
        telemetry=TelemetryConfig(**dict(cfg_any.telemetry)),
        execution=ExecutionConfig(**dict(cfg_any.execution)),
        benchmark=BenchmarkConfig(**dict(cfg_any.benchmark)),
    )


def main() -> None:
    """CLI entrypoint. Imports Hydra lazily so tests need not install it."""

    import hydra
    from omegaconf import DictConfig

    config_path = str(Path(__file__).resolve().parents[3] / "conf")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="config")
    def _hydra_main(cfg: DictConfig) -> None:
        config = _from_omegaconf(cfg)
        results = run_experiment(config)
        for result in results:
            print(result["generated_text"])
            if config.telemetry.pretty_report:
                print_telemetry_report(result["telemetry"])

    _hydra_main()


if __name__ == "__main__":
    main()
