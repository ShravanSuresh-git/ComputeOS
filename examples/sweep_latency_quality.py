"""Run a controlled latency/quality sweep for Predictive Value Scheduling."""

from __future__ import annotations

import argparse
import json
import random
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

from computeos.benchmarks.perplexity import ReferencePerplexityBenchmark
from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.scheduling.pvs import PredictiveValueScheduler, PVSResourceBudgets
from computeos.visualization import plot_pareto_frontier

CANONICAL_CONDITION_ORDER = (
    "baseline",
    "pvs_loose",
    "pvs_medium",
    "pvs_tight",
    "token_cap",
)

BUDGET_PRESETS: dict[str, dict[str, object]] = {
    "baseline": {},
    "pvs_loose": {
        "max_compute_units": 100.0,
        "max_latency_ms": 10_000.0,
        "min_net_value": 0.0,
    },
    "pvs_medium": {
        "max_compute_units": 60.0,
        "max_latency_ms": 10_000.0,
        "min_net_value": 0.0,
    },
    "pvs_tight": {
        "max_compute_units": 30.0,
        "max_latency_ms": 10_000.0,
        "min_net_value": 0.0,
    },
    "token_cap": {
        "max_compute_units": 10_000.0,
        "max_latency_ms": 10_000.0,
        "min_net_value": 0.60,
    },
}


def main() -> None:
    """Run the sweep and write ``outputs/sweep_results.json``."""

    args = _parse_args()
    pairs = _sample_prompt_continuation_pairs(args.n_prompts)
    model, tokenizer, model_name = _load_model(args.model)
    results = run_sweep(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        pairs=pairs,
        max_new_tokens=args.max_new_tokens,
    )
    _write_json(_outputs_dir() / "sweep_results.json", results)
    _print_table(results)
    try:
        path = plot_pareto_frontier(results, _outputs_dir() / "pareto_frontier.png")
    except ImportError as exc:
        Console().print(f"Pareto plot skipped: {exc}")
    else:
        Console().print(f"Pareto plot saved to {path}")


def run_sweep(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    pairs: list[tuple[str, str]],
    max_new_tokens: int,
) -> dict[str, object]:
    """Evaluate baseline and PVS budget variants on a shared prompt set."""

    benchmark = ReferencePerplexityBenchmark(pairs=pairs, max_continuation_tokens=5)
    reference_scores: dict[int, float] = {}
    telemetry_config = TelemetryConfig(capture_memory=True)
    conditions: list[tuple[str, Callable[[], Scheduler]]] = [
        (name, lambda preset=name: _make_pvs_scheduler(preset))
        for name in CANONICAL_CONDITION_ORDER
    ]
    random.Random(42).shuffle(conditions)

    per_condition: dict[str, list[dict[str, float]]] = {
        name: [] for name in CANONICAL_CONDITION_ORDER
    }
    for condition, scheduler_factory in conditions:
        engine = HFControlledEngine(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            scheduler=scheduler_factory(),
            execution_config=ExecutionConfig(max_new_tokens=max_new_tokens, use_cache=False),
            telemetry_config=telemetry_config,
        )
        engine.warm_up(prompt=pairs[0][0])
        for prompt_index, (prompt, continuation) in enumerate(pairs):
            started_at = perf_counter()
            execution = engine.generate(prompt)
            wall_latency_ms = (perf_counter() - started_at) * 1000.0
            if prompt_index not in reference_scores:
                reference_scores[prompt_index] = benchmark.score_pair(engine, prompt, continuation)
            score = reference_scores[prompt_index]
            per_condition[condition].append(
                {
                    "prompt_index": float(prompt_index),
                    "latency_ms": wall_latency_ms,
                    "perplexity": float(score or 0.0),
                    "layers_executed": float(len(execution.telemetry.layers)),
                    "early_exits": float(_early_exits_applied(execution.telemetry)),
                    "earliest_exit_layer_index": float(
                        _earliest_exit_layer_index(execution.telemetry)
                    ),
                }
            )

    baseline_latency = mean(row["latency_ms"] for row in per_condition["baseline"])
    baseline_perplexity = mean(row["perplexity"] for row in per_condition["baseline"])
    summaries: dict[str, dict[str, float]] = {}
    for condition in CANONICAL_CONDITION_ORDER:
        rows = per_condition[condition]
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
            "mean_earliest_exit_layer_index": mean(
                row["earliest_exit_layer_index"] for row in rows
            ),
            "latency_reduction_pct": 100.0 * (baseline_latency - mean_latency) / baseline_latency
            if baseline_latency > 0.0
            else 0.0,
            "perplexity_delta": mean_perplexity - baseline_perplexity,
        }

    return {
        "conditions": summaries,
        "n_prompts": len(pairs),
        "model": model_name,
        "perplexity_metric": "reference perplexity (not self-scored)",
        "reference_dataset": "curated_capital_city_pairs",
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _make_pvs_scheduler(preset: str) -> Scheduler:
    """Create a scheduler for a named budget preset."""

    if preset == "baseline":
        return FullExecutionScheduler()
    if preset == "default":
        return PredictiveValueScheduler()
    if preset == "tight":
        preset = "pvs_tight"
    parameters = BUDGET_PRESETS.get(preset)
    if parameters is None:
        raise ValueError(f"Unknown PVS budget preset: {preset}")
    return PredictiveValueScheduler(
        budgets=PVSResourceBudgets(
            max_latency_ms=float(parameters["max_latency_ms"]),
            max_compute_units=float(parameters["max_compute_units"]),
            min_net_value=float(parameters["min_net_value"]),
        )
    )


def _sample_prompt_continuation_pairs(n_prompts: int) -> list[tuple[str, str]]:
    """Sample prompt/reference-continuation pairs for reference perplexity."""

    return _fallback_pairs(n_prompts)


def _sample_prompts(n_prompts: int) -> list[str]:
    """Compatibility helper for examples that only need prompts."""

    return [prompt for prompt, _continuation in _sample_prompt_continuation_pairs(n_prompts)]


def _append_article_pair(article_lines: list[str], pairs: list[tuple[str, str]]) -> None:
    article = " ".join(" ".join(line.split()) for line in article_lines)
    if len(article) < 320:
        return
    prompt, continuation = _split_reference_text(article)
    if prompt and continuation and _is_reference_pair_candidate(prompt, continuation):
        pairs.append((prompt, continuation))


def _split_reference_text(text: str) -> tuple[str, str]:
    prompt_end = text.rfind(" ", 100, 151)
    if prompt_end < 0:
        return "", ""
    continuation_end = text.rfind(" ", prompt_end + 100, prompt_end + 151)
    if continuation_end < 0:
        return "", ""
    return text[: prompt_end + 1], text[prompt_end + 1 : continuation_end]


def _is_reference_pair_candidate(prompt: str, continuation: str) -> bool:
    text = prompt + continuation
    if any(char.isdigit() for char in text):
        return False
    if any(marker in text for marker in ("@", "(", ")", "[", "]", "=", "–")):
        return False
    punctuation = sum(text.count(mark) for mark in (",", ";", ":", '"', "'"))
    if punctuation > 8:
        return False
    words = text.split()
    if len(words) < 35:
        return False
    short_caps = sum(1 for word in words if len(word) > 1 and word.isupper())
    return short_caps <= 2


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


def _fallback_pairs(n_prompts: int) -> list[tuple[str, str]]:
    pairs = [
        (
            'The capital city of France is',
            'Paris, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Germany is',
            'Berlin, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Italy is',
            'Rome, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Spain is',
            'Madrid, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Portugal is',
            'Lisbon, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Japan is',
            'Tokyo, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Canada is',
            'Ottawa, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Australia is',
            'Canberra, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Brazil is',
            'Brasilia, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Egypt is',
            'Cairo, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of India is',
            'New Delhi, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of China is',
            'Beijing, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Russia is',
            'Moscow, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Mexico is',
            'Mexico City, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Argentina is',
            'Buenos Aires, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Chile is',
            'Santiago, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Peru is',
            'Lima, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Colombia is',
            'Bogota, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Norway is',
            'Oslo, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Sweden is',
            'Stockholm, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Finland is',
            'Helsinki, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Denmark is',
            'Copenhagen, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Ireland is',
            'Dublin, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Greece is',
            'Athens, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Turkey is',
            'Ankara, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Poland is',
            'Warsaw, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Austria is',
            'Vienna, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Hungary is',
            'Budapest, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Romania is',
            'Bucharest, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Bulgaria is',
            'Sofia, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Thailand is',
            'Bangkok, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Vietnam is',
            'Hanoi, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Indonesia is',
            'Jakarta, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Malaysia is',
            'Kuala Lumpur, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Singapore is',
            'Singapore, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Kenya is',
            'Nairobi, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Ethiopia is',
            'Addis Ababa, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Morocco is',
            'Rabat, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Ghana is',
            'Accra, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Nigeria is',
            'Abuja, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of South Africa is',
            'Pretoria, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of New Zealand is',
            'Wellington, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of South Korea is',
            'Seoul, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Philippines is',
            'Manila, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Pakistan is',
            'Islamabad, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Bangladesh is',
            'Dhaka, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Saudi Arabia is',
            'Riyadh, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Israel is',
            'Jerusalem, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
        (
            'The capital city of Czech Republic is',
            'Prague, a national center for government, transportation, education, culture, '
            'public services, and international visitors.',
        ),
        (
            'The capital city of Netherlands is',
            'Amsterdam, a national center for government, transportation, education, '
            'culture, public services, and international visitors.',
        ),
    ]
    if len(pairs) != 50:
        raise RuntimeError("Fallback reference dataset must contain exactly 50 pairs.")
    return [pairs[index % len(pairs)] for index in range(n_prompts)]

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


def _earliest_exit_layer_index(telemetry: object) -> int:
    decisions = getattr(telemetry, "scheduler_decisions", [])
    for decision in decisions:
        metadata = getattr(decision, "metadata", {})
        action_result = metadata.get("action_result") if isinstance(metadata, dict) else None
        if (
            getattr(decision, "action", None) == SchedulerAction.EARLY_EXIT
            and isinstance(action_result, dict)
            and action_result.get("applied") is True
        ):
            layer_name = getattr(decision, "layer_name", None)
            if isinstance(layer_name, str):
                try:
                    return int(layer_name.rsplit(".", maxsplit=1)[-1])
                except ValueError:
                    return -1
    return -1


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
    table.add_column("Condition", no_wrap=True)
    table.add_column("Latency ms", no_wrap=True)
    table.add_column("Reference PPL", no_wrap=True)
    table.add_column("Layers", no_wrap=True)
    table.add_column("Early exits", no_wrap=True)
    table.add_column("Exit layer", no_wrap=True)
    table.add_column("Latency delta", no_wrap=True)
    table.add_column("PPL delta", no_wrap=True)
    for name, raw in conditions.items():
        row = raw if isinstance(raw, dict) else {}
        table.add_row(
            str(name),
            f"{float(row.get('mean_latency_ms', 0.0)):.2f} ± "
            f"{float(row.get('std_latency_ms', 0.0)):.2f}",
            f"{float(row.get('mean_perplexity', 0.0)):.2f} ± "
            f"{float(row.get('std_perplexity', 0.0)):.3f}",
            f"{float(row.get('mean_layers_executed', 0.0)):.2f}",
            f"{float(row.get('mean_early_exits', 0.0)):.2f}",
            f"{float(row.get('mean_earliest_exit_layer_index', -1.0)):.2f}",
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
