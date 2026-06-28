from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.controlled import ControlledForwardRuntime, RuntimeBudget
from computeos.execution.engine import InferenceEngine
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


class AlwaysExitScheduler(Scheduler):
    def reset(self) -> None:
        pass

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        return SchedulerDecision(
            action=SchedulerAction.EARLY_EXIT,
            layer_name=context.layer_name,
            reason="test early exit",
        )


class FakeTokenizer:
    eos_token_id: int | None = None

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class FakeCausalLM(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        logits = torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32)
        logits[:, -1, 2] = 10.0
        return SimpleNamespace(logits=logits)


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

    def test_inference_engine_early_exit_applied(self) -> None:
        engine = InferenceEngine(
            model=FakeCausalLM(),
            tokenizer=FakeTokenizer(),
            model_name="fake",
            scheduler=AlwaysExitScheduler(),
            execution_config=ExecutionConfig(max_new_tokens=4),
            telemetry_config=TelemetryConfig(),
        )

        result = engine.generate("hello")

        self.assertTrue(result.telemetry.metadata["early_exit_applied"])
        self.assertEqual(result.telemetry.metadata["tokens_generated"], 1)


if __name__ == "__main__":
    unittest.main()
