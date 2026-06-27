"""Minimal prompt benchmark for smoke testing the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from computeos.benchmarks.base import Benchmark, BenchmarkItem


@dataclass(frozen=True)
class PromptSmokeBenchmark(Benchmark):
    """Simple prompt list benchmark."""

    prompts: list[str]
    limit: int | None = None

    def items(self) -> list[BenchmarkItem]:
        prompts = self.prompts[: self.limit] if self.limit is not None else self.prompts
        return [BenchmarkItem(prompt=prompt) for prompt in prompts]
