from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from computeos.benchmarks.base import BenchmarkItem, BenchmarkResult
from computeos.benchmarks.controlled import ControlledBenchmarkRunner
from computeos.benchmarks.perplexity import PerplexityBenchmark, ReferencePerplexityBenchmark
from computeos.benchmarks.registry import default_benchmark_registry
from computeos.benchmarks.reporting import export_benchmark_report, rows_from_results
from computeos.benchmarks.wikitext import WikitextPerplexityBenchmark
from computeos.config.schema import BenchmarkConfig, ExecutionConfig, TelemetryConfig
from computeos.execution.controlled import ControlledForwardRuntime
from computeos.execution.engine import ExecutionResult
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerDecision
from computeos.scheduling.random_scheduler import RandomScheduler
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


class TinyTokenizer:
    eos_token_id: int | None = None

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        token_ids = [ord(char) % 31 + 1 for char in prompt] or [1]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class FullExecutionScheduler(Scheduler):
    def reset(self) -> None:
        pass

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        return SchedulerDecision.record_only("test full execution")


class BenchmarkTests(unittest.TestCase):
    def _tiny_hf_engine(self) -> HFControlledEngine:
        torch.manual_seed(11)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=64,
                n_layer=2,
                n_head=2,
                n_embd=64,
                n_positions=64,
                n_ctx=64,
            )
        )
        return HFControlledEngine(
            model=model,
            tokenizer=TinyTokenizer(),  # type: ignore[arg-type]
            model_name="random-gpt2",
            scheduler=FullExecutionScheduler(),
            execution_config=ExecutionConfig(max_new_tokens=2, seed=11, use_cache=False),
            telemetry_config=TelemetryConfig(capture_memory=False),
        )

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

    def test_controlled_benchmark_runner_produces_results(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        runtime = ControlledForwardRuntime(
            model=model,
            scheduler=RandomScheduler(exit_prob=0.0),
        )
        benchmark = PerplexityBenchmark(prompts=["test"])

        results = ControlledBenchmarkRunner(runtime, benchmark).run()

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0].metadata["layers_executed"], 0)

    def test_reference_perplexity_higher_than_one(self) -> None:
        engine = self._tiny_hf_engine()
        benchmark = ReferencePerplexityBenchmark(pairs=[("hello ", "world")])

        score = benchmark.score_pair(engine, "hello ", "world")

        self.assertGreater(score, 1.0)

    def test_score_continuation_length(self) -> None:
        engine = self._tiny_hf_engine()

        scores = engine.score_continuation("ab", "cde", max_tokens=2)

        self.assertEqual(len(scores), 2)

    def test_self_scored_perplexity_is_not_reference_perplexity(self) -> None:
        engine = self._tiny_hf_engine()
        prompt = "hello "
        execution = engine.generate(prompt)
        self_scored = PerplexityBenchmark(prompts=[prompt]).score(
            BenchmarkItem(prompt=prompt),
            execution,
        )
        reference = ReferencePerplexityBenchmark(pairs=[(prompt, "world")]).score_pair(
            engine,
            prompt,
            "world",
        )

        self.assertIsNotNone(self_scored)
        self.assertGreater(abs(float(self_scored or 0.0) - reference), 0.5)


if __name__ == "__main__":
    unittest.main()
