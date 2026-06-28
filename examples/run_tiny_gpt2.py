"""Run a tiny ComputeOS smoke experiment.

This example downloads `sshleifer/tiny-gpt2`, runs a short generation, and
prints the telemetry that matters when validating a new scheduler or runtime.
"""

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


def main() -> None:
    config = ComputeOSConfig(
        model=ModelConfig(name="sshleifer/tiny-gpt2"),
        scheduler=SchedulerConfig(
            name="heuristic",
            parameters={"confidence_threshold": 0.85, "entropy_threshold": 1.5},
        ),
        telemetry=TelemetryConfig(
            enabled=True,
            capture_activations=True,
            capture_attention_entropy=True,
            capture_memory=True,
            log_to_wandb=False,
        ),
        execution=ExecutionConfig(max_new_tokens=8, use_cache=True, seed=17),
        benchmark=BenchmarkConfig(
            name="prompt_smoke",
            prompts=["ComputeOS allocates inference compute by"],
            limit=1,
        ),
    )

    result = run_experiment(config)[0]
    telemetry = result["telemetry"]

    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['generated_text']}")
    print_telemetry_report(telemetry, max_layers=8)


if __name__ == "__main__":
    main()
