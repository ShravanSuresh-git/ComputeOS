"""ComputeOS v1.0 deterministic release demo.

This demo avoids large model downloads. It proves the core release path:
controlled runtime enforcement, telemetry, replay, counterfactual analysis, and
publication-style report export.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from computeos.execution.controlled import ControlledForwardRuntime
from computeos.replay import CounterfactualEngine, CounterfactualScenario, ScenarioType, TraceLoader
from computeos.replay.experiment import CounterfactualExperiment
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision


class DemoScheduler(Scheduler):
    """Small scheduler that requests an early exit after one executed layer."""

    def reset(self) -> None:
        pass

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        if context.metadata.get("decision_point") == "pre_layer":
            return SchedulerDecision(
                action=SchedulerAction.CONTINUE,
                layer_name=context.layer_name,
                reason="demo pre-layer continue",
            )
        if context.layer_telemetry is not None and context.layer_name == "0":
            return SchedulerDecision(
                action=SchedulerAction.EARLY_EXIT,
                layer_name=context.layer_name,
                reason="demo early exit after first layer",
            )
        return SchedulerDecision.record_only("demo record")


def main() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    result = ControlledForwardRuntime(model, DemoScheduler(), model_name="demo").run(torch.ones(1, 4))
    trace = TraceLoader().from_telemetry(result.telemetry)
    scenario = CounterfactualScenario(
        name="continue_one_more_layer",
        scenario_type=ScenarioType.CONTINUE_EXTRA_LAYERS,
        extra_layers=1,
    )
    counterfactual = CounterfactualEngine().evaluate(trace, scenario)
    output_dir = Path("outputs/demo")
    paths = CounterfactualExperiment().export(
        CounterfactualExperiment().default_policy_comparison(trace),
        output_dir,
    )

    print("ComputeOS release demo")
    print(f"Output shape: {tuple(result.output.shape)}")
    print(f"Executed layers: {len(result.telemetry.layers)}")
    print(f"Applied actions: {len(result.action_results)}")
    print(f"Counterfactual utility: {counterfactual.predicted_utility:.6f}")
    print(f"Reports written to: {output_dir}")
    print(", ".join(f"{name}={path.name}" for name, path in paths.items()))


if __name__ == "__main__":
    main()
