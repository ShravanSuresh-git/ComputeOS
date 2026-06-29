"""Adapter for running benchmarks through ControlledForwardRuntime."""

from __future__ import annotations

import torch

from computeos.benchmarks.base import Benchmark, BenchmarkResult
from computeos.execution.controlled import ControlledForwardRuntime
from computeos.execution.engine import ExecutionResult


class ControlledBenchmarkRunner:
    """Run a benchmark through ControlledForwardRuntime instead of InferenceEngine."""

    def __init__(
        self,
        runtime: ControlledForwardRuntime,
        benchmark: Benchmark,
    ) -> None:
        self._runtime = runtime
        self._benchmark = benchmark

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for item in self._benchmark.items():
            controlled_result = self._runtime.run(_controlled_input(item.prompt))
            execution = ExecutionResult(
                prompt=item.prompt,
                generated_text=str(controlled_result.output),
                telemetry=controlled_result.telemetry,
                raw_outputs={
                    "action_results": [
                        {
                            "action": action_result.requested_action,
                            "applied": action_result.applied,
                            "reason": action_result.reason,
                        }
                        for action_result in controlled_result.action_results
                    ]
                },
            )
            results.append(
                BenchmarkResult(
                    item=item,
                    execution=execution,
                    score=self._benchmark.score(item, execution),
                    metadata={
                        "actions_applied": sum(
                            1
                            for action_result in controlled_result.action_results
                            if action_result.applied
                        ),
                        "layers_executed": len(controlled_result.telemetry.layers),
                    },
                )
            )
        return results


def _controlled_input(prompt: str) -> object:
    return torch.ones(1, 4)
