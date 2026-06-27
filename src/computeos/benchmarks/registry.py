"""Benchmark registry."""

from __future__ import annotations

from collections.abc import Callable

from computeos.benchmarks.base import Benchmark
from computeos.benchmarks.prompt_smoke import PromptSmokeBenchmark
from computeos.config.schema import BenchmarkConfig

BenchmarkFactory = Callable[[BenchmarkConfig], Benchmark]


class BenchmarkRegistry:
    """Explicit benchmark registry for experiment runners."""

    def __init__(self) -> None:
        self._factories: dict[str, BenchmarkFactory] = {}

    def register(self, name: str, factory: BenchmarkFactory) -> None:
        if not name:
            raise ValueError("Benchmark name must be non-empty.")
        self._factories[name] = factory

    def create(self, config: BenchmarkConfig) -> Benchmark:
        try:
            factory = self._factories[config.name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown benchmark '{config.name}'. Available: {available}") from exc
        return factory(config)


def _prompt_smoke_factory(config: BenchmarkConfig) -> Benchmark:
    return PromptSmokeBenchmark(prompts=config.prompts, limit=config.limit)


def default_benchmark_registry() -> BenchmarkRegistry:
    registry = BenchmarkRegistry()
    registry.register("prompt_smoke", _prompt_smoke_factory)
    return registry
