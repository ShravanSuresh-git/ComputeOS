from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.config.schema import SchedulerConfig
from computeos.execution.engine import ExecutionResult
from computeos.experiments.comparison import PolicyComparisonRunner
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.registry import default_scheduler_registry
from computeos.telemetry.metrics import ActivationStats, LayerTelemetry, ModelTelemetry


class FakeComparisonEngine:
    def __init__(self) -> None:
        self._scheduler = default_scheduler_registry().create(SchedulerConfig(name="heuristic"))
        self._execution_config = SimpleNamespace(n_runs=1, seed=42)

    def generate(self, prompt: str) -> ExecutionResult:
        telemetry = ModelTelemetry(model_name="fake")
        telemetry.layers.append(
            LayerTelemetry(
                layer_name="layer.0",
                layer_type="Linear",
                latency_ms=1.0,
                activation_stats=ActivationStats(
                    mean=0.0,
                    std=0.1,
                    min=-1.0,
                    max=1.0,
                    l2_norm=1.0,
                    numel=16,
                ),
            )
        )
        telemetry.metadata["tokens_generated"] = 1
        telemetry.metadata["log_prob_per_token"] = [-0.5]
        telemetry.total_latency_ms = 1.0
        decision = self._scheduler.decide(
            SchedulerContext(
                step_index=0,
                layer_name="layer.0",
                layer_telemetry=telemetry.layers[0],
                model_telemetry=telemetry,
            )
        )
        telemetry.scheduler_decisions.append(decision)
        return ExecutionResult(
            prompt=prompt,
            generated_text="generated",
            telemetry=telemetry,
            raw_outputs={},
        )


class ComparisonTests(unittest.TestCase):
    def test_policy_comparison_rows_and_csv(self) -> None:
        registry = default_scheduler_registry()
        schedulers = [
            ("heuristic", registry.create(SchedulerConfig(name="heuristic"))),
            ("random", registry.create(SchedulerConfig(name="random"))),
        ]
        benchmark = PerplexityBenchmark(prompts=["a", "b"])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            report = PolicyComparisonRunner(
                schedulers=schedulers,
                benchmark=benchmark,
                engine=FakeComparisonEngine(),
                output_dir=output_dir,
            ).run()

            self.assertEqual(len(report.rows), 4)
            csv_path = output_dir / "comparison.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertIn("scheduler", rows[0])
            self.assertIn("normalized_regret", rows[0])

    def test_n_runs_adds_std_fields(self) -> None:
        registry = default_scheduler_registry()
        engine = FakeComparisonEngine()
        engine._execution_config = SimpleNamespace(n_runs=2, seed=42)
        report = PolicyComparisonRunner(
            schedulers=[("heuristic", registry.create(SchedulerConfig(name="heuristic")))],
            benchmark=PerplexityBenchmark(prompts=["a"]),
            engine=engine,
        ).run()

        self.assertIn("score_std", report.rows[0])
        self.assertIn("latency_std_ms", report.rows[0])


if __name__ == "__main__":
    unittest.main()
