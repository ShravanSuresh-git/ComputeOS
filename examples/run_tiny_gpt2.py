"""Run a tiny ComputeOS smoke experiment.

This example downloads `sshleifer/tiny-gpt2`, runs a short generation, and
prints the telemetry that matters when validating a new scheduler or runtime.
"""

from __future__ import annotations

from statistics import mean

from computeos.config.schema import (
    BenchmarkConfig,
    ComputeOSConfig,
    ExecutionConfig,
    ModelConfig,
    SchedulerConfig,
    TelemetryConfig,
)
from computeos.experiments.runner import run_experiment


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
    latencies = [layer.latency_ms for layer in telemetry.layers]

    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['generated_text']}")
    print(f"Total latency: {telemetry.total_latency_ms:.3f} ms")
    print(f"Layer events: {len(telemetry.layers)}")
    print(f"Scheduler decisions: {len(telemetry.scheduler_decisions)}")
    print(f"Mean layer latency: {mean(latencies):.3f} ms")
    print(f"Peak process RSS: {telemetry.peak_process_rss_bytes} bytes")


if __name__ == "__main__":
    main()
