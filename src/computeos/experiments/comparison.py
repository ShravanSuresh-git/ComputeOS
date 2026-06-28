"""Policy comparison runner for scheduler benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from computeos.benchmarks.base import Benchmark
from computeos.execution.engine import InferenceEngine
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
        for name, scheduler in self._schedulers:
            original = self._engine._scheduler
            self._engine._scheduler = scheduler
            scheduler.reset()
            try:
                results = self._benchmark.run(self._engine)
            finally:
                self._engine._scheduler = original

            for result in results:
                telemetry = result.execution.telemetry
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

                early_exits = sum(
                    1
                    for decision in telemetry.scheduler_decisions
                    if decision.action == SchedulerAction.EARLY_EXIT
                )
                row: dict[str, Any] = {
                    "scheduler": name,
                    "prompt": result.item.prompt,
                    "score": result.score,
                    "latency_ms": telemetry.total_latency_ms,
                    "tokens_generated": telemetry.metadata.get("tokens_generated", 0),
                    "compute_units": telemetry.metadata.get("compute_units", 0.0),
                    "early_exits": early_exits,
                    "sequence_regret": regret.sequence_regret,
                    "normalized_regret": regret.normalized_regret,
                }
                row.update(
                    {
                        key: value
                        for key, value in (result.metadata or {}).items()
                        if isinstance(value, (int, float, str))
                    }
                )
                report.rows.append(row)

        if self._output_dir is not None:
            report.to_csv(self._output_dir / "comparison.csv")
        return report
