from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from computeos.config.schema import TelemetryConfig
from computeos.instrumentation.hooks import HookedTransformerMonitor
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerDecision
from computeos.telemetry.collector import TelemetryCollector


class RecordingScheduler(Scheduler):
    def __init__(self) -> None:
        self.contexts: list[SchedulerContext] = []

    def reset(self) -> None:
        self.contexts.clear()

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        self.contexts.append(context)
        return SchedulerDecision.record_only()


class TinyTransformerLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([nn.Linear(4, 4), nn.ReLU()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.transformer.h:
            x = layer(x)
        return x


class HookTests(unittest.TestCase):
    def test_hooks_record_layer_telemetry_and_scheduler_decisions(self) -> None:
        model = TinyTransformerLikeModel()
        scheduler = RecordingScheduler()
        collector = TelemetryCollector(model_name="tiny")
        with HookedTransformerMonitor(
            model=model,
            scheduler=scheduler,
            collector=collector,
            telemetry_config=TelemetryConfig(capture_memory=False),
        ):
            output = model(torch.ones(2, 4))

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(len(collector.model_telemetry.layers), 2)
        self.assertEqual(len(collector.model_telemetry.scheduler_decisions), 2)
        self.assertEqual(len(scheduler.contexts), 2)
        self.assertIsNotNone(collector.model_telemetry.layers[0].activation_stats)


if __name__ == "__main__":
    unittest.main()
