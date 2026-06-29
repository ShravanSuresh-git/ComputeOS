"""Validate CRI predictions against controlled full-inference reruns."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

from rich.console import Console
from sweep_latency_quality import FullExecutionScheduler, _load_model, _sample_prompts

from computeos.benchmarks.base import BenchmarkItem
from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.replay.counterfactual_engine import CounterfactualEngine
from computeos.replay.scenario import CounterfactualScenario, ScenarioType
from computeos.replay.trace_loader import TraceLoader
from computeos.scheduling.pvs import PredictiveValueScheduler, PVSResourceBudgets


def main() -> None:
    """Run CRI validation and write ``outputs/cri_validation.json``."""

    args = _parse_args()
    prompts = _sample_prompts(args.n_prompts)
    model, tokenizer, model_name = _load_model(args.model)
    benchmark = PerplexityBenchmark(prompts=prompts)
    telemetry_config = TelemetryConfig(capture_memory=True)
    trace_loader = TraceLoader()
    cri = CounterfactualEngine()
    scenario = CounterfactualScenario(
        name="continue_all_static",
        scenario_type=ScenarioType.REPLACE_SCHEDULER,
        scheduler_name="static",
    )
    per_prompt: list[dict[str, float | int]] = []

    for prompt_index, prompt in enumerate(prompts):
        pvs_engine = HFControlledEngine(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            scheduler=PredictiveValueScheduler(
                budgets=PVSResourceBudgets(
                    max_latency_ms=250.0,
                    max_compute_units=256.0,
                    min_net_value=0.0,
                )
            ),
            execution_config=ExecutionConfig(max_new_tokens=args.max_new_tokens, use_cache=False),
            telemetry_config=telemetry_config,
        )
        pvs_execution = pvs_engine.generate(prompt)
        trace = trace_loader.from_telemetry(pvs_execution.telemetry)
        predicted = cri.evaluate(trace, scenario)

        baseline_engine = HFControlledEngine(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            scheduler=FullExecutionScheduler(),
            execution_config=ExecutionConfig(max_new_tokens=args.max_new_tokens, use_cache=False),
            telemetry_config=telemetry_config,
        )
        started_at = perf_counter()
        baseline_execution = baseline_engine.generate(prompt)
        actual_latency_ms = (perf_counter() - started_at) * 1000.0
        actual_perplexity = float(
            benchmark.score(BenchmarkItem(prompt=prompt), baseline_execution) or 0.0
        )
        cri_predicted_perplexity = _perplexity_proxy(predicted.predicted_quality_proxy)
        per_prompt.append(
            {
                "prompt_index": prompt_index,
                "cri_predicted_latency_ms": predicted.predicted_latency_ms,
                "actual_latency_ms": actual_latency_ms,
                "cri_predicted_perplexity": cri_predicted_perplexity,
                "actual_perplexity": actual_perplexity,
            }
        )

    latency_mae = mean(
        abs(float(row["cri_predicted_latency_ms"]) - float(row["actual_latency_ms"]))
        for row in per_prompt
    )
    perplexity_mae = mean(
        abs(float(row["cri_predicted_perplexity"]) - float(row["actual_perplexity"]))
        for row in per_prompt
    )
    payload: dict[str, object] = {
        "n_prompts": len(prompts),
        "latency_mae_ms": latency_mae,
        "perplexity_mae": perplexity_mae,
        "per_prompt": per_prompt,
        "model": model_name,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    path = _outputs_dir() / "cri_validation.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latency_pass = latency_mae < 20.0
    perplexity_pass = perplexity_mae < 2.0
    Console().print(
        "CRI validation "
        f"{'PASS' if latency_pass and perplexity_pass else 'FAIL'}: "
        f"latency MAE={latency_mae:.2f} ms "
        f"({'pass' if latency_pass else 'fail'}, threshold <20 ms), "
        f"perplexity MAE={perplexity_mae:.3f} "
        f"({'pass' if perplexity_pass else 'fail'}, threshold <2.0). "
        "The static counterfactual uses observed telemetry only, so high error "
        "identifies where learned CRI calibration is still needed."
    )


def _perplexity_proxy(quality_proxy: float) -> float:
    clipped = min(0.999, max(0.001, quality_proxy))
    return 1.0 / clipped


def _outputs_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--model", type=str, default="distilgpt2")
    return parser.parse_args()


if __name__ == "__main__":
    main()
