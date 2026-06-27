"""Benchmark abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from computeos.execution.engine import ExecutionResult, InferenceEngine


@dataclass(frozen=True)
class BenchmarkItem:
    """One benchmark example."""

    prompt: str
    expected: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    """Benchmark output for one item."""

    item: BenchmarkItem
    execution: ExecutionResult
    score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class Benchmark(ABC):
    """Benchmark contract for standard and custom evaluation tasks."""

    @abstractmethod
    def items(self) -> list[BenchmarkItem]:
        """Return benchmark items to execute."""

    def run(self, engine: InferenceEngine) -> list[BenchmarkResult]:
        """Run all benchmark items through an inference engine."""

        results: list[BenchmarkResult] = []
        for item in self.items():
            execution = engine.generate(item.prompt)
            results.append(
                BenchmarkResult(
                    item=item,
                    execution=execution,
                    score=self.score(item, execution),
                )
            )
        return results

    def score(self, item: BenchmarkItem, execution: ExecutionResult) -> float | None:
        """Score an execution result. Override for benchmark-specific metrics."""

        return None
