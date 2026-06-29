"""Policy comparison runner for scheduler benchmarks."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from computeos.benchmarks.base import Benchmark, BenchmarkResult
from computeos.execution.engine import InferenceEngine
from computeos.experiments.pareto import ParetoPoint, pareto_frontier
from computeos.replay.oracle_scheduler import OracleObjective, OracleScheduler
from computeos.replay.regret import SchedulerRegret, compute_regret
from computeos.replay.trace_loader import TraceLoader
from computeos.scheduling.base import Scheduler
from computeos.scheduling.decision import SchedulerAction


@dataclass
class ComparisonReport:
    """Flat scheduler comparison rows and regret summaries."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    regret_by_scheduler: dict[str, SchedulerRegret] = field(default_factory=dict)
    pareto_points: list[ParetoPoint] = field(default_factory=list)
    all_points: list[ParetoPoint] = field(default_factory=list)

    def to_csv(self, path: Path) -> None:
        if not self.rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


class PolicyComparisonRunner:
    """Run multiple schedulers against one benchmark and compare regret."""

    def __init__(
        self,
        schedulers: list[tuple[str, Scheduler]],
        benchmark: Benchmark,
        engine: InferenceEngine,
        output_dir: Path | None = None,
    ) -> None:
        self._schedulers = schedulers
        self._benchmark = benchmark
        self._engine = engine
        self._output_dir = output_dir
        self._oracle = OracleScheduler()
        self._loader = TraceLoader()

    def run(self) -> ComparisonReport:
        report = ComparisonReport()
        n_runs = max(
            1,
            int(getattr(getattr(self._engine, "_execution_config", None), "n_runs", 1)),
        )
        for name, scheduler in self._schedulers:
            original = self._engine._scheduler
            self._engine._scheduler = scheduler
            scheduler.reset()
            try:
                all_run_results: list[list[BenchmarkResult]] = []
                for run_index in range(n_runs):
                    seed = int(
                        getattr(
                            getattr(self._engine, "_execution_config", None),
                            "seed",
                            42,
                        )
                    )
                    torch.manual_seed(seed + run_index)
                    all_run_results.append(self._benchmark.run(self._engine))
            finally:
                self._engine._scheduler = original

            for item_results in zip(*all_run_results, strict=True):
                representative = item_results[0]
                telemetry = representative.execution.telemetry
                trace = self._loader.from_telemetry(telemetry)
                oracle_plan = self._oracle.plan(trace, objective=OracleObjective.MAXIMIZE_UTILITY)
                oracle_utils = [decision.utility for decision in oracle_plan.decisions]
                online_utils: list[float] = []
                for decision in telemetry.scheduler_decisions:
                    prediction = decision.metadata.get("prediction")
                    if isinstance(prediction, dict):
                        online_utils.append(float(prediction.get("expected_net_value", 0.0)))
                    else:
                        online_utils.append(0.0)
                regret = compute_regret(oracle_utils, online_utils)
                report.regret_by_scheduler[name] = regret
                scores = [result.score for result in item_results if result.score is not None]
                latencies = [
                    result.execution.telemetry.total_latency_ms
                    for result in item_results
                    if result.execution.telemetry.total_latency_ms is not None
                ]

                early_exits = sum(
                    1
                    for decision in telemetry.scheduler_decisions
                    if decision.action == SchedulerAction.EARLY_EXIT
                )
                row: dict[str, Any] = {
                    "scheduler": name,
                    "prompt": representative.item.prompt,
                    "score": representative.score,
                    "latency_ms": telemetry.total_latency_ms,
                    "score_mean": statistics.mean(scores) if scores else None,
                    "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "latency_mean_ms": statistics.mean(latencies) if latencies else None,
                    "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                    "tokens_generated": telemetry.metadata.get("tokens_generated", 0),
                    "compute_units": telemetry.metadata.get("compute_units", 0.0),
                    "early_exits": early_exits,
                    "sequence_regret": regret.sequence_regret,
                    "normalized_regret": regret.normalized_regret,
                }
                row.update(
                    {
                        key: value
                        for key, value in (representative.metadata or {}).items()
                        if isinstance(value, (int, float, str))
                    }
                )
                report.rows.append(row)

        report.all_points = [
            ParetoPoint(
                scheduler=str(row.get("scheduler", "")),
                latency_ms=float(row["latency_ms"]),
                score=float(row["score"]),
                row=row,
            )
            for row in report.rows
            if row.get("latency_ms") is not None and row.get("score") is not None
        ]
        report.pareto_points = pareto_frontier(report.rows)
        if self._output_dir is not None:
            report.to_csv(self._output_dir / "comparison.csv")
            from computeos.experiments.artifacts import ArtifactStore

            try:
                store = ArtifactStore(output_dir=self._output_dir)
                store.snapshot_report(report)
            except Exception:
                pass
        return report
