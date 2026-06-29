"""Measure oracle gap for controlled ComputeOS scheduling conditions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import torch
from rich.console import Console
from rich.table import Table
from sweep_latency_quality import (
    CANONICAL_CONDITION_ORDER,
    _load_model,
    _make_pvs_scheduler,
    _sample_prompt_continuation_pairs,
)

from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.replay.oracle_scheduler import OracleObjective, OracleScheduler
from computeos.replay.trace_loader import TraceLoader, quality_proxy, trace_utility


@dataclass(frozen=True)
class OracleGapRow:
    """Aggregated oracle-gap metrics for one scheduler condition."""

    mean_oracle_gap: float
    oracle_efficiency_pct: float
    mean_achieved_utility: float
    mean_oracle_utility: float


def main() -> None:
    """Run oracle-gap measurement and write ``outputs/oracle_gap.json``."""

    args = _parse_args()
    pairs = _sample_prompt_continuation_pairs(args.n_prompts)
    model, tokenizer, model_name = _load_model(args.model)
    telemetry_config = TelemetryConfig(capture_memory=True)
    trace_loader = TraceLoader()
    oracle = OracleScheduler()
    condition_names = [
        name for name in CANONICAL_CONDITION_ORDER if name != "token_cap"
    ]
    rows: dict[str, OracleGapRow] = {}

    for condition in condition_names:
        gaps: list[float] = []
        achieved_utilities: list[float] = []
        oracle_utilities: list[float] = []
        engine = HFControlledEngine(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            scheduler=_make_pvs_scheduler(condition),
            execution_config=ExecutionConfig(max_new_tokens=args.max_new_tokens, use_cache=False),
            telemetry_config=telemetry_config,
        )
        engine.warm_up(prompt=pairs[0][0])
        for prompt, _continuation in pairs:
            execution = engine.generate(prompt)
            trace = trace_loader.from_telemetry(execution.telemetry)
            plan = oracle.plan(trace, objective=OracleObjective.BALANCED)
            achieved_utility = trace_utility(
                latency_ms=trace.total_latency_ms,
                memory_mb=trace.peak_memory_mb,
                compute_units=trace.total_compute_units,
                quality_proxy=quality_proxy(execution.telemetry),
            )
            gap = _bounded_gap(plan.utility, achieved_utility)
            gaps.append(gap)
            achieved_utilities.append(achieved_utility)
            oracle_utilities.append(plan.utility)

        mean_gap = mean(gaps)
        rows[condition] = OracleGapRow(
            mean_oracle_gap=mean_gap,
            oracle_efficiency_pct=100.0 * (1.0 - mean_gap),
            mean_achieved_utility=mean(achieved_utilities),
            mean_oracle_utility=mean(oracle_utilities),
        )

    payload = {
        "conditions": {
            condition: {
                "mean_oracle_gap": row.mean_oracle_gap,
                "oracle_efficiency_pct": row.oracle_efficiency_pct,
                "mean_achieved_utility": row.mean_achieved_utility,
                "mean_oracle_utility": row.mean_oracle_utility,
            }
            for condition, row in rows.items()
        },
        "n_prompts": len(pairs),
        "model": model_name,
        "objective": str(OracleObjective.BALANCED),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    output = _outputs_dir() / "oracle_gap.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_table(rows)


def _bounded_gap(oracle_utility: float, achieved_utility: float) -> float:
    raw_gap = (oracle_utility - achieved_utility) / (abs(oracle_utility) + 1e-9)
    return min(1.0, max(0.0, raw_gap))


def _outputs_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _print_table(rows: dict[str, OracleGapRow]) -> None:
    table = Table(title="ComputeOS Oracle Gap")
    table.add_column("Condition", no_wrap=True)
    table.add_column("Oracle efficiency (%)", no_wrap=True)
    table.add_column("Mean achieved utility", no_wrap=True)
    table.add_column("Mean oracle utility", no_wrap=True)
    table.add_column("Oracle gap", no_wrap=True)
    for condition, row in rows.items():
        table.add_row(
            condition,
            f"{row.oracle_efficiency_pct:.1f}",
            f"{row.mean_achieved_utility:.4f}",
            f"{row.mean_oracle_utility:.4f}",
            f"{row.mean_oracle_gap:.3f}",
        )
    Console().print(table)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--model", type=str, default="distilgpt2")
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(17)
    main()
