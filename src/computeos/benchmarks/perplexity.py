"""Prompt perplexity benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

from computeos.benchmarks.base import Benchmark, BenchmarkItem, BenchmarkResult
from computeos.execution.engine import ExecutionResult, InferenceEngine
from computeos.scheduling.decision import SchedulerAction


@dataclass(frozen=True)
class PerplexityBenchmark(Benchmark):
    """Compute perplexity from per-token log probabilities in telemetry metadata."""

    prompts: list[str]
    limit: int | None = None

    def items(self) -> list[BenchmarkItem]:
        prompts = self.prompts[: self.limit] if self.limit is not None else self.prompts
        return [BenchmarkItem(prompt=prompt) for prompt in prompts]

    def score(self, item: BenchmarkItem, execution: ExecutionResult) -> float | None:
        log_probs_raw = execution.telemetry.metadata.get("log_prob_per_token", [])
        if not isinstance(log_probs_raw, list) or not log_probs_raw:
            return None
        log_probs = [float(value) for value in log_probs_raw]
        return math.exp(-sum(log_probs) / len(log_probs))

    def run(self, engine: InferenceEngine) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for item in self.items():
            execution = engine.generate(item.prompt)
            results.append(
                BenchmarkResult(
                    item=item,
                    execution=execution,
                    score=self.score(item, execution),
                    metadata={
                        "tokens_generated": execution.telemetry.metadata.get(
                            "tokens_generated",
                            0,
                        ),
                        "latency_ms": execution.telemetry.total_latency_ms,
                        "compute_units": execution.telemetry.metadata.get(
                            "compute_units",
                            0.0,
                        ),
                        "early_exits_applied": _early_exits_applied(execution),
                    },
                )
            )
        return results


def _early_exits_applied(execution: ExecutionResult) -> int:
    count = 0
    for decision in execution.telemetry.scheduler_decisions:
        action_result = decision.metadata.get("action_result")
        if (
            decision.action == SchedulerAction.EARLY_EXIT
            and isinstance(action_result, dict)
            and action_result.get("applied") is True
        ):
            count += 1
    return count
