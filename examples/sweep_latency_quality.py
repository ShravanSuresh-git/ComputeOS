"""Run a controlled latency/quality sweep for Predictive Value Scheduling."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import torch
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from computeos.benchmarks.base import BenchmarkItem
from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.scheduling.pvs import PredictiveValueScheduler, PVSResourceBudgets


def main() -> None:
    """Run the sweep and write ``outputs/sweep_results.json``."""

    args = _parse_args()
    prompts = _sample_prompts(args.n_prompts)
    model, tokenizer, model_name = _load_model(args.model)
    results = run_sweep(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
    )
    _write_json(_outputs_dir() / "sweep_results.json", results)
    _print_table(results)


def run_sweep(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    prompts: list[str],
    max_new_tokens: int,
) -> dict[str, object]:
    """Evaluate baseline and PVS budget variants on a shared prompt set."""

    benchmark = PerplexityBenchmark(prompts=prompts)
    telemetry_config = TelemetryConfig(capture_memory=True)
    conditions: dict[str, Callable[[], Scheduler]] = {
        "baseline": FullExecutionScheduler,
        "pvs_loose": lambda: PredictiveValueScheduler(
            budgets=PVSResourceBudgets(
                max_latency_ms=500.0,
                max_compute_units=512.0,
                min_net_value=-0.1,
            )
        ),
        "pvs_medium": lambda: PredictiveValueScheduler(
            budgets=PVSResourceBudgets(
                max_latency_ms=250.0,
                max_compute_units=256.0,
                min_net_value=0.0,
            )
        ),
        "pvs_tight": lambda: PredictiveValueScheduler(
            budgets=PVSResourceBudgets(
                max_latency_ms=100.0,
                max_compute_units=128.0,
                min_net_value=0.05,
            )
        ),
    }

    per_condition: dict[str, list[dict[str, float]]] = {name: [] for name in conditions}
    for condition, scheduler_factory in conditions.items():
        engine = HFControlledEngine(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            scheduler=scheduler_factory(),
            execution_config=ExecutionConfig(max_new_tokens=max_new_tokens, use_cache=False),
            telemetry_config=telemetry_config,
        )
        for prompt_index, prompt in enumerate(prompts):
            started_at = perf_counter()
            execution = engine.generate(prompt)
            wall_latency_ms = (perf_counter() - started_at) * 1000.0
            score = benchmark.score(BenchmarkItem(prompt=prompt), execution)
            per_condition[condition].append(
                {
                    "prompt_index": float(prompt_index),
                    "latency_ms": wall_latency_ms,
                    "perplexity": float(score or 0.0),
                    "layers_executed": float(len(execution.telemetry.layers)),
                    "early_exits": float(_early_exits_applied(execution.telemetry)),
                }
            )

    baseline_latency = mean(row["latency_ms"] for row in per_condition["baseline"])
    baseline_perplexity = mean(row["perplexity"] for row in per_condition["baseline"])
    summaries: dict[str, dict[str, float]] = {}
    for condition, rows in per_condition.items():
        latency_values = [row["latency_ms"] for row in rows]
        perplexity_values = [row["perplexity"] for row in rows]
        mean_latency = mean(latency_values)
        mean_perplexity = mean(perplexity_values)
        summaries[condition] = {
            "mean_latency_ms": mean_latency,
            "std_latency_ms": _std(latency_values),
            "mean_perplexity": mean_perplexity,
            "std_perplexity": _std(perplexity_values),
            "mean_layers_executed": mean(row["layers_executed"] for row in rows),
            "mean_early_exits": mean(row["early_exits"] for row in rows),
            "latency_reduction_pct": 100.0 * (baseline_latency - mean_latency) / baseline_latency
            if baseline_latency > 0.0
            else 0.0,
            "perplexity_delta": mean_perplexity - baseline_perplexity,
        }

    return {
        "conditions": summaries,
        "n_prompts": len(prompts),
        "model": model_name,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _sample_prompts(n_prompts: int) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        return _fallback_prompts(n_prompts)

    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    except Exception:
        return _fallback_prompts(n_prompts)

    prompts: list[str] = []
    for row in dataset:
        text = str(row.get("text", ""))
        if "\n = " in text or text.startswith(" = "):
            trimmed = " ".join(text.split())[:200]
            if trimmed:
                prompts.append(trimmed)
        if len(prompts) >= n_prompts:
            break
    return prompts or _fallback_prompts(n_prompts)


class FullExecutionScheduler(Scheduler):
    """Scheduler baseline that never requests adaptive runtime actions."""

    def reset(self) -> None:
        """No state is maintained between prompts."""

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        """Record the decision point and continue full execution."""

        return SchedulerDecision(
            action=SchedulerAction.RECORD_ONLY,
            layer_name=context.layer_name,
            reason="full execution baseline",
        )


def _fallback_prompts(n_prompts: int) -> list[str]:
    seeds = [
        "ComputeOS studies adaptive inference scheduling for transformer models.",
        "Runtime telemetry can expose when additional layers have diminishing returns.",
        "A research framework should make scheduling policies easy to compare.",
        "Counterfactual replay estimates what alternative compute plans might have done.",
        "Predictive value scheduling treats inference as an optimal stopping problem.",
    ]
    return [seeds[index % len(seeds)] for index in range(n_prompts)]


def _load_model(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer, model_name


def _early_exits_applied(telemetry: object) -> int:
    decisions = getattr(telemetry, "scheduler_decisions", [])
    count = 0
    for decision in decisions:
        metadata = getattr(decision, "metadata", {})
        action_result = metadata.get("action_result") if isinstance(metadata, dict) else None
        if (
            getattr(decision, "action", None) == SchedulerAction.EARLY_EXIT
            and isinstance(action_result, dict)
            and action_result.get("applied") is True
        ):
            count += 1
    return count


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _outputs_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_table(results: dict[str, object]) -> None:
    conditions = results["conditions"]
    if not isinstance(conditions, dict):
        return
    table = Table(title="ComputeOS PVS Latency/Quality Sweep")
    table.add_column("Condition")
    table.add_column("Latency ms")
    table.add_column("PPL")
    table.add_column("Layers")
    table.add_column("Early exits")
    table.add_column("Latency delta")
    table.add_column("PPL delta")
    for name, raw in conditions.items():
        row = raw if isinstance(raw, dict) else {}
        table.add_row(
            str(name),
            f"{float(row.get('mean_latency_ms', 0.0)):.2f} +/- "
            f"{float(row.get('std_latency_ms', 0.0)):.2f}",
            f"{float(row.get('mean_perplexity', 0.0)):.3f} +/- "
            f"{float(row.get('std_perplexity', 0.0)):.3f}",
            f"{float(row.get('mean_layers_executed', 0.0)):.2f}",
            f"{float(row.get('mean_early_exits', 0.0)):.2f}",
            f"{float(row.get('latency_reduction_pct', 0.0)):.1f}%",
            f"{float(row.get('perplexity_delta', 0.0)):.3f}",
        )
    Console().print(table)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--model", type=str, default="distilgpt2")
    return parser.parse_args()


if __name__ == "__main__":
    main()
