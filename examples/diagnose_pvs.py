"""Inspect Predictive Value Scheduling internals for one controlled generation."""

from __future__ import annotations

import argparse
import math

from rich.console import Console
from rich.table import Table
from sweep_latency_quality import _early_exits_applied, _load_model, _make_pvs_scheduler

from computeos.benchmarks.base import BenchmarkItem
from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.instrumentation.layers import discover_transformer_layers
from computeos.scheduling.decision import SchedulerAction
from computeos.scheduling.pvs import PredictiveValueScheduler


def main() -> None:
    """Run a single prompt and print PVS value/cost diagnostics."""

    args = _parse_args()
    model, tokenizer, model_name = _load_model(args.model)
    scheduler = _make_pvs_scheduler(args.budget_preset)
    if not isinstance(scheduler, PredictiveValueScheduler):
        raise TypeError("diagnose_pvs.py requires a PredictiveValueScheduler preset.")
    engine = HFControlledEngine(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        scheduler=scheduler,
        execution_config=ExecutionConfig(max_new_tokens=args.max_new_tokens, use_cache=False),
        telemetry_config=TelemetryConfig(capture_memory=True),
    )
    execution = engine.generate(args.prompt)
    trace = scheduler.replay()

    table = Table(title=f"PVS Diagnostic: {args.budget_preset}", expand=False)
    for column in (
        "token",
        "layer",
        "entropy",
        "uncertainty",
        "act_norm",
        "layer_var",
        "exp_improvement",
        "exp_cost",
        "net_value",
        "cum_compute",
        "action",
    ):
        table.add_column(column, no_wrap=True)

    for event in trace:
        action = event.action
        action_label = action.upper()
        if action == SchedulerAction.EARLY_EXIT:
            action_label = f"[red]{action_label}[/red]"
        elif action == SchedulerAction.SKIP_LAYER:
            action_label = f"[yellow]{action_label}[/yellow]"
        table.add_row(
            str(event.step_index),
            _truncate(event.layer_name or "", 20),
            f"{event.features.entropy:.2f}",
            f"{event.features.uncertainty:.2f}",
            f"{event.features.activation_norm:.2f}",
            f"{event.features.layer_variance:.2f}",
            f"{event.prediction.expected_improvement:.3f}",
            f"{event.prediction.expected_cost:.3f}",
            f"{event.prediction.expected_net_value:.3f}",
            f"{event.cumulative_compute_units:.1f}",
            action_label,
        )

    console = Console(width=180)
    console.print(table)
    final_perplexity = PerplexityBenchmark(prompts=[args.prompt]).score(
        BenchmarkItem(prompt=args.prompt),
        execution,
    )
    layers_executed = len(execution.telemetry.layers)
    total_possible = _total_possible_layers(model, args.max_new_tokens)
    early_exits = _early_exits_applied(execution.telemetry)
    console.print(
        f"{layers_executed} / {total_possible} layers executed, "
        f"{early_exits} early exits, final_perplexity={float(final_perplexity or math.nan):.3f}"
    )
    if early_exits == 0:
        net_values = [event.prediction.expected_net_value for event in trace]
        final_compute = trace[-1].cumulative_compute_units if trace else 0.0
        console.print(
            "why it didn't exit: "
            f"mean_net_value={_mean(net_values):.3f}, "
            f"max_net_value={max(net_values, default=0.0):.3f}, "
            f"final_compute={final_compute:.1f} / "
            f"{scheduler.budgets.max_compute_units:.1f} budget"
        )


def _total_possible_layers(model: object, max_new_tokens: int) -> int:
    return len(discover_transformer_layers(model)) * max_new_tokens


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument(
        "--prompt",
        type=str,
        default="The history of artificial intelligence begins with",
    )
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--budget-preset",
        choices=("default", "tight", "token_cap"),
        default="default",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
