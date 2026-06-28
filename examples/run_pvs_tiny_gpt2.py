"""Run Predictive Value Scheduling on tiny GPT-2."""

from __future__ import annotations

from computeos.config.schema import (
    BenchmarkConfig,
    ComputeOSConfig,
    ExecutionConfig,
    ModelConfig,
    SchedulerConfig,
    TelemetryConfig,
)
from computeos.experiments.runner import run_experiment
from computeos.telemetry.reports import print_telemetry_report
from computeos.visualization.pvs import extract_pvs_trace, plot_pvs_trace


def main() -> None:
    config = ComputeOSConfig(
        model=ModelConfig(name="sshleifer/tiny-gpt2"),
        scheduler=SchedulerConfig(
            name="pvs",
            parameters={
                "budgets": {
                    "max_latency_ms": 100.0,
                    "max_memory_mb": 1024.0,
                    "max_compute_units": 128.0,
                    "min_net_value": 0.0,
                }
            },
        ),
        telemetry=TelemetryConfig(capture_memory=True, log_to_wandb=False),
        execution=ExecutionConfig(max_new_tokens=8, use_cache=True, seed=17),
        benchmark=BenchmarkConfig(
            name="prompt_smoke",
            prompts=["ComputeOS allocates inference compute by"],
            limit=1,
        ),
    )

    result = run_experiment(config)[0]
    telemetry = result["telemetry"]
    trace = extract_pvs_trace(telemetry)

    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['generated_text']}")
    print_telemetry_report(telemetry, max_layers=8)
    print("\nPVS replay first 5 decisions")
    for record in trace[:5]:
        prediction = record["prediction"]
        print(
            f"{record['index']}: {record['action']} "
            f"net={prediction['expected_net_value']:.4f} "
            f"reason={record['reason']}"
        )

    try:
        path = plot_pvs_trace(telemetry, "outputs/pvs_timeline.png")
    except ImportError:
        print("\nInstall matplotlib to generate outputs/pvs_timeline.png")
    else:
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
