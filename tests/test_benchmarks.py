from __future__ import annotations

import unittest

from computeos.config.schema import BenchmarkConfig
from computeos.benchmarks.registry import default_benchmark_registry


class BenchmarkTests(unittest.TestCase):
    def test_prompt_smoke_items_respect_limit(self) -> None:
        benchmark = default_benchmark_registry().create(
            BenchmarkConfig(name="prompt_smoke", prompts=["a", "b", "c"], limit=2)
        )
        self.assertEqual([item.prompt for item in benchmark.items()], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
