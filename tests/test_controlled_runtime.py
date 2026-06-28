from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from computeos.execution.controlled import ControlledForwardRuntime, RuntimeBudget
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision


class CountingLayer(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return inputs + self.increment


class DecisionScheduler(Scheduler):
    def __init__(self, decisions: dict[tuple[str, str], SchedulerAction]) -> None:
        self.decisions = decisions

    def reset(self) -> None:
        pass

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        point = str(context.metadata.get("decision_point", "post_layer"))
        action = self.decisions.get((point, context.layer_name or ""), SchedulerAction.CONTINUE)
        return SchedulerDecision(
            action=action,
            layer_name=context.layer_name,
            reason=f"{point}:{action}",
        )


class ControlledRuntimeTests(unittest.TestCase):
    def test_early_exit_changes_execution(self) -> None:
        first = CountingLayer(1.0)
        second = CountingLayer(10.0)
        model = nn.Sequential(first, second)
        scheduler = DecisionScheduler({("post_layer", "0"): SchedulerAction.EARLY_EXIT})

        result = ControlledForwardRuntime(model, scheduler).run(torch.tensor([0.0]))

        self.assertEqual(float(result.output.item()), 1.0)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)
        self.assertEqual(result.action_results[0].requested_action, SchedulerAction.EARLY_EXIT)
        self.assertTrue(
            result.telemetry.scheduler_decisions[-1].metadata["action_result"]["applied"]
        )

    def test_skip_layer_changes_execution(self) -> None:
        first = CountingLayer(1.0)
        second = CountingLayer(10.0)
        model = nn.Sequential(first, second)
        scheduler = DecisionScheduler({("pre_layer", "1"): SchedulerAction.SKIP_LAYER})

        result = ControlledForwardRuntime(model, scheduler).run(torch.tensor([0.0]))

        self.assertEqual(float(result.output.item()), 1.0)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)
        self.assertEqual(result.action_results[0].requested_action, SchedulerAction.SKIP_LAYER)

    def test_layer_budget_is_enforced(self) -> None:
        model = nn.Sequential(CountingLayer(1.0), CountingLayer(1.0), CountingLayer(1.0))
        scheduler = DecisionScheduler({})

        result = ControlledForwardRuntime(
            model,
            scheduler,
            budget=RuntimeBudget(max_layers=1),
        ).run(torch.tensor([0.0]))

        self.assertEqual(float(result.output.item()), 1.0)
        self.assertEqual(len(result.telemetry.layers), 1)
        self.assertEqual(result.action_results[0].reason, "max layer budget reached")


if __name__ == "__main__":
    unittest.main()
