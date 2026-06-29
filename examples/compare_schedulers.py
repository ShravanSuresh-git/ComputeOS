"""Compare all schedulers on tiny-gpt2 using a shared engine."""

from __future__ import annotations

from pathlib import Path

from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.config.schema import ExecutionConfig, ModelConfig, SchedulerConfig, TelemetryConfig
from computeos.execution.engine import InferenceEngine
from computeos.execution.model_loader import load_hf_causal_lm
from computeos.experiments.comparison import PolicyComparisonRunner
from computeos.scheduling.registry import default_scheduler_registry


def main() -> None:
    model_config = ModelConfig(name="sshleifer/tiny-gpt2")
    loaded = load_hf_causal_lm(model_config)
    execution_config = ExecutionConfig(max_new_tokens=16, seed=42)
    telemetry_config = TelemetryConfig()

    registry = default_scheduler_registry()
    scheduler_names = ["heuristic", "pvs", "entropy", "confidence", "random"]
    schedulers = [
        (name, registry.create(SchedulerConfig(name=name)))
        for name in scheduler_names
    ]

    _, placeholder_scheduler = schedulers[0]
    engine = InferenceEngine(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        model_name=loaded.name,
        scheduler=placeholder_scheduler,
        execution_config=execution_config,
        telemetry_config=telemetry_config,
    )

    benchmark = PerplexityBenchmark(
        prompts=["The future of inference is", "Adaptive scheduling allows"],
        limit=2,
    )

    runner = PolicyComparisonRunner(
        schedulers=schedulers,
        benchmark=benchmark,
        engine=engine,
        output_dir=Path("outputs"),
    )
    report = runner.run()
    for row in report.rows:
        print(row)

    if report.pareto_points:
        print("\nPareto-optimal schedulers (quality vs latency):")
        for point in report.pareto_points:
            print(
                f"  {point.scheduler}: "
                f"latency={point.latency_ms:.1f}ms  perplexity={point.score:.4f}"
            )
        from computeos.experiments.pareto import plot_pareto

        plot_pareto(report.pareto_points, report.all_points, Path("outputs/pareto.png"))
        print("Pareto plot saved to outputs/pareto.png")


if __name__ == "__main__":
    main()
