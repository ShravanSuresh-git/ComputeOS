"""Contract tests: every registered scheduler must satisfy the Scheduler API."""

from __future__ import annotations

import unittest

from computeos.config.schema import SchedulerConfig
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.registry import default_scheduler_registry
from computeos.telemetry.metrics import ModelTelemetry


class TestSchedulerContract(unittest.TestCase):
    def _minimal_context(self) -> SchedulerContext:
        return SchedulerContext(
            step_index=0,
            layer_name=None,
            layer_telemetry=None,
            model_telemetry=ModelTelemetry(model_name="contract_test"),
        )

    def test_all_registered_schedulers_satisfy_contract(self) -> None:
        registry = default_scheduler_registry()
        for name in registry.names():
            with self.subTest(scheduler=name):
                scheduler = registry.create(SchedulerConfig(name=name))
                scheduler.reset()
                decision = scheduler.decide(self._minimal_context())
                self.assertIsNotNone(decision.action, f"{name}: action must not be None")
                self.assertIsInstance(decision.reason, str, f"{name}: reason must be a str")
                self.assertGreater(len(decision.reason), 0, f"{name}: reason must be non-empty")


if __name__ == "__main__":
    unittest.main()
