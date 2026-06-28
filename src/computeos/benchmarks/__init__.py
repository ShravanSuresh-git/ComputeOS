"""Benchmark interfaces and implementations."""

from computeos.benchmarks.base import Benchmark, BenchmarkItem, BenchmarkResult
from computeos.benchmarks.prompt_smoke import PromptSmokeBenchmark
from computeos.benchmarks.registry import BenchmarkRegistry, default_benchmark_registry
from computeos.benchmarks.wikitext import WikitextPerplexityBenchmark

__all__ = [
    "Benchmark",
    "BenchmarkItem",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "PromptSmokeBenchmark",
    "WikitextPerplexityBenchmark",
    "default_benchmark_registry",
]
