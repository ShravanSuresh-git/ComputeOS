from __future__ import annotations

import math
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from computeos.benchmarks.base import BenchmarkItem
from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision


class TinyTokenizer:
    eos_token_id: int | None = None

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        token_ids = [ord(char) % 31 + 1 for char in prompt][:8] or [1]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class LayerActionScheduler(Scheduler):
    def __init__(self, actions: dict[str, SchedulerAction]) -> None:
        self._actions = actions

    def reset(self) -> None:
        pass

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        action = self._actions.get(context.layer_name or "", SchedulerAction.CONTINUE)
        return SchedulerDecision(
            action=action,
            layer_name=context.layer_name,
            reason=f"test {action}",
        )


class HFControlledEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=64,
                n_layer=2,
                n_head=2,
                n_embd=64,
                n_positions=32,
                n_ctx=32,
            )
        )
        self.tokenizer = TinyTokenizer()

    def _engine(self, scheduler: Scheduler) -> HFControlledEngine:
        return HFControlledEngine(
            model=self.model,
            tokenizer=self.tokenizer,  # type: ignore[arg-type]
            model_name="random-gpt2",
            scheduler=scheduler,
            execution_config=ExecutionConfig(max_new_tokens=1, seed=7, use_cache=False),
            telemetry_config=TelemetryConfig(capture_memory=False),
        )

    def test_skip_layer_changes_output(self) -> None:
        baseline = self._engine(LayerActionScheduler({})).generate("hello")
        skipped = self._engine(
            LayerActionScheduler({"transformer.h.0": SchedulerAction.SKIP_LAYER})
        ).generate("hello")

        baseline_scores = baseline.raw_outputs["scores"][0]
        skipped_scores = skipped.raw_outputs["scores"][0]
        self.assertFalse(torch.allclose(baseline_scores, skipped_scores))
        self.assertEqual(len(skipped.telemetry.layers), 1)
        self.assertEqual(skipped.telemetry.layers[0].layer_name, "transformer.h.1")

    def test_early_exit_before_full_forward(self) -> None:
        result = self._engine(
            LayerActionScheduler({"transformer.h.1": SchedulerAction.EARLY_EXIT})
        ).generate("hello")

        self.assertTrue(result.telemetry.metadata["early_exit_applied"])
        self.assertEqual(result.telemetry.metadata["early_exit_layer"], "transformer.h.1")
        self.assertEqual(len(result.telemetry.layers), 1)
        self.assertEqual(result.telemetry.layers[0].layer_name, "transformer.h.0")

    def test_telemetry_layers_contains_exactly_layers_that_ran(self) -> None:
        result = self._engine(LayerActionScheduler({})).generate("hello")

        self.assertEqual(
            [layer.layer_name for layer in result.telemetry.layers],
            ["transformer.h.0", "transformer.h.1"],
        )

    def test_perplexity_is_finite_positive_after_early_exit(self) -> None:
        result = self._engine(
            LayerActionScheduler({"transformer.h.1": SchedulerAction.EARLY_EXIT})
        ).generate("hello")
        score = PerplexityBenchmark(prompts=["hello"]).score(BenchmarkItem(prompt="hello"), result)

        self.assertIsNotNone(score)
        self.assertTrue(math.isfinite(score or 0.0))
        self.assertGreater(score or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
