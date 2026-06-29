"""Calibrate Predictive Value Scheduling weights from oracle replay traces."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from rich.console import Console
from rich.table import Table
from sweep_latency_quality import _load_model, _sample_prompts

from computeos.benchmarks.base import BenchmarkItem
from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.replay.trace_loader import TraceLoader
from computeos.scheduling.pvs import (
    PredictiveValueScheduler,
    PVSCostWeights,
    PVSResourceBudgets,
    PVSValueWeights,
    calibrate_weights,
)


def main() -> None:
    """Collect oracle traces, fit PVS weights, and rerun medium budgets."""

    args = _parse_args()
    prompts = _sample_prompts(args.n_prompts)
    model, tokenizer, model_name = _load_model(args.model)
    telemetry_config = TelemetryConfig(capture_memory=True)
    execution_config = ExecutionConfig(max_new_tokens=args.max_new_tokens, use_cache=False)
    trace_loader = TraceLoader()
    traces = []

    for prompt in prompts:
        engine = HFControlledEngine(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            scheduler=PredictiveValueScheduler(
                budgets=PVSResourceBudgets(
                    max_latency_ms=1_000_000.0,
                    max_compute_units=1_000_000.0,
                    min_net_value=-999.0,
                )
            ),
            execution_config=execution_config,
            telemetry_config=telemetry_config,
        )
        execution = engine.generate(prompt)
        traces.append(trace_loader.from_telemetry(execution.telemetry))

    fitted = calibrate_weights(traces, objective="maximize_utility")
    default = PVSValueWeights()
    default_metrics = _run_medium(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        execution_config=execution_config,
        telemetry_config=telemetry_config,
        value_weights=default,
    )
    calibrated_metrics = _run_medium(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        execution_config=execution_config,
        telemetry_config=telemetry_config,
        value_weights=fitted,
    )
    payload: dict[str, object] = {
        "fitted_weights": asdict(fitted),
        "default_weights": asdict(default),
        "cost_weights": asdict(PVSCostWeights()),
        "pvs_medium_default": default_metrics,
        "pvs_medium_calibrated": calibrated_metrics,
        "n_prompts": len(prompts),
        "model": model_name,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    path = _outputs_dir() / "calibrated_weights.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_table(fitted, default, default_metrics, calibrated_metrics)


def _run_medium(
    prompts: list[str],
    model: object,
    tokenizer: object,
    model_name: str,
    execution_config: ExecutionConfig,
    telemetry_config: TelemetryConfig,
    value_weights: PVSValueWeights,
) -> dict[str, float]:
    benchmark = PerplexityBenchmark(prompts=prompts)
    latencies: list[float] = []
    perplexities: list[float] = []
    for prompt in prompts:
        engine = HFControlledEngine(
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,  # type: ignore[arg-type]
            model_name=model_name,
            scheduler=PredictiveValueScheduler(
                budgets=PVSResourceBudgets(
                    max_latency_ms=250.0,
                    max_compute_units=256.0,
                    min_net_value=0.0,
                ),
                value_weights=value_weights,
            ),
            execution_config=execution_config,
            telemetry_config=telemetry_config,
        )
        started_at = perf_counter()
        execution = engine.generate(prompt)
        latencies.append((perf_counter() - started_at) * 1000.0)
        perplexities.append(float(benchmark.score(BenchmarkItem(prompt=prompt), execution) or 0.0))
    return {
        "mean_latency_ms": mean(latencies),
        "mean_perplexity": mean(perplexities),
    }


def _print_table(
    fitted: PVSValueWeights,
    default: PVSValueWeights,
    default_metrics: dict[str, float],
    calibrated_metrics: dict[str, float],
) -> None:
    table = Table(title="PVS Oracle Calibration")
    table.add_column("Weight")
    table.add_column("Default")
    table.add_column("Fitted")
    for key, default_value in asdict(default).items():
        table.add_row(key, f"{float(default_value):.4f}", f"{float(asdict(fitted)[key]):.4f}")
    Console().print(table)
    Console().print(
        "Medium PVS default: "
        f"{default_metrics['mean_latency_ms']:.2f} ms, "
        f"PPL {default_metrics['mean_perplexity']:.3f}"
    )
    Console().print(
        "Medium PVS calibrated: "
        f"{calibrated_metrics['mean_latency_ms']:.2f} ms, "
        f"PPL {calibrated_metrics['mean_perplexity']:.3f}"
    )


def _outputs_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--model", type=str, default="distilgpt2")
    return parser.parse_args()


if __name__ == "__main__":
    main()
