from __future__ import annotations

import math
import unittest

from computeos.benchmarks.base import BenchmarkItem, BenchmarkResult
from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.benchmarks.registry import default_benchmark_registry
from computeos.benchmarks.reporting import export_benchmark_report, rows_from_results
from computeos.benchmarks.wikitext import WikitextPerplexityBenchmark
from computeos.config.schema import BenchmarkConfig
from computeos.execution.engine import ExecutionResult
from computeos.telemetry.metrics import ModelTelemetry


class MockPerplexityEngine:
    def generate(self, prompt: str) -> ExecutionResult:
        telemetry = ModelTelemetry(model_name="mock")
        telemetry.metadata["log_prob_per_token"] = [-0.5, -0.5, -0.5]
        telemetry.metadata["tokens_generated"] = 3
        return ExecutionResult(
            prompt=prompt,
            generated_text="generated",
            telemetry=telemetry,
            raw_outputs={},
        )


class BenchmarkTests(unittest.TestCase):
    def test_prompt_smoke_items_respect_limit(self) -> None:
        benchmark = default_benchmark_registry().create(
            BenchmarkConfig(name="prompt_smoke", prompts=["a", "b", "c"], limit=2)
        )
        self.assertEqual([item.prompt for item in benchmark.items()], ["a", "b"])

    def test_wikitext_adapter_is_registered_without_loading_dataset(self) -> None:
        benchmark = default_benchmark_registry().create(
            BenchmarkConfig(
                name="wikitext_perplexity",
                limit=3,
                parameters={"dataset_config": "wikitext-2-raw-v1"},
            )
        )
        self.assertIsInstance(benchmark, WikitextPerplexityBenchmark)

    def test_benchmark_report_exports_all_publication_formats(self) -> None:
        import tempfile

        telemetry = ModelTelemetry(model_name="tiny", total_latency_ms=1.0)
        result = BenchmarkResult(
            item=BenchmarkItem(prompt="hello"),
            execution=ExecutionResult(
                prompt="hello",
                generated_text="hello world",
                telemetry=telemetry,
                raw_outputs={},
            ),
            score=0.5,
        )
        rows = rows_from_results([result])
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = export_benchmark_report(rows, temp_dir)

            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["latex"].exists())
            self.assertTrue(paths["html"].exists())

    def test_benchmark_report_handles_empty_rows(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = export_benchmark_report([], temp_dir)

            self.assertIn("prompt", paths["csv"].read_text(encoding="utf-8"))
            self.assertEqual(paths["json"].read_text(encoding="utf-8"), "[]")

    def test_benchmark_report_escapes_table_content(self) -> None:
        import tempfile

        telemetry = ModelTelemetry(model_name="tiny", total_latency_ms=1.0)
        result = BenchmarkResult(
            item=BenchmarkItem(prompt="hello | world"),
            execution=ExecutionResult(
                prompt="hello | world",
                generated_text="<tag> & value",
                telemetry=telemetry,
                raw_outputs={},
            ),
            score=0.5,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = export_benchmark_report(rows_from_results([result]), temp_dir)

            self.assertIn("hello \\| world", paths["markdown"].read_text(encoding="utf-8"))
            self.assertIn("&lt;tag&gt; &amp; value", paths["html"].read_text(encoding="utf-8"))
            self.assertIn(r"\&", paths["latex"].read_text(encoding="utf-8"))

    def test_perplexity_benchmark_score(self) -> None:
        benchmark = PerplexityBenchmark(prompts=["hello"])

        result = benchmark.run(MockPerplexityEngine())[0]

        self.assertAlmostEqual(result.score or 0.0, math.exp(0.5), places=4)
        self.assertEqual(result.metadata["tokens_generated"], 3)


if __name__ == "__main__":
    unittest.main()
