from __future__ import annotations

import unittest

from computeos.benchmarks.wikitext import WikitextPerplexityBenchmark
from computeos.benchmarks.registry import default_benchmark_registry
from computeos.config.schema import BenchmarkConfig


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


if __name__ == "__main__":
    unittest.main()
