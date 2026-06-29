from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from computeos.replay import (
    CounterfactualEngine,
    CounterfactualScenario,
    OracleObjective,
    OracleScheduler,
    ScenarioType,
    TraceLoader,
    TracePlayer,
)
from computeos.replay.experiment import CounterfactualExperiment
from computeos.replay.regret import compute_regret
from computeos.replay.trace_loader import RuntimeEventType
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.telemetry.metrics import ActivationStats, LayerTelemetry, ModelTelemetry


def _telemetry() -> ModelTelemetry:
    telemetry = ModelTelemetry(model_name="tiny")
    for index in range(3):
        telemetry.layers.append(
            LayerTelemetry(
                layer_name=f"transformer.h.{index}",
                layer_type="GPT2Block",
                latency_ms=1.0 + index,
                activation_stats=ActivationStats(
                    mean=0.0,
                    std=0.1,
                    min=-1.0,
                    max=1.0,
                    l2_norm=2.0,
                    numel=1024,
                ),
                process_rss_bytes=(100 + index) * 1024 * 1024,
            )
        )
        telemetry.scheduler_decisions.append(
            SchedulerDecision(
                action=SchedulerAction.EARLY_EXIT if index == 1 else SchedulerAction.CONTINUE,
                layer_name=f"transformer.h.{index}",
                confidence=0.8,
                reason="test",
                metadata={
                    "prediction": {
                        "expected_improvement": 0.3,
                        "expected_net_value": 0.2 - 0.1 * index,
                    }
                },
            )
        )
    telemetry.confidence_scores.extend([0.6, 0.7])
    telemetry.total_latency_ms = 6.0
    telemetry.peak_process_rss_bytes = 103 * 1024 * 1024
    return telemetry


class ReplayCRITests(unittest.TestCase):
    def test_trace_loader_creates_replay_events(self) -> None:
        trace = TraceLoader().from_telemetry(_telemetry())

        self.assertEqual(trace.model_name, "tiny")
        self.assertEqual(trace.events[0].event_type, RuntimeEventType.REQUEST_STARTED)
        self.assertEqual(trace.events[-1].event_type, RuntimeEventType.REQUEST_FINISHED)
        self.assertEqual(len(trace.layer_events()), 3)
        self.assertEqual(len(trace.decision_events()), 3)

    def test_trace_player_pause_resume_step_seek(self) -> None:
        player = TracePlayer(TraceLoader().from_telemetry(_telemetry()))

        self.assertTrue(player.state.paused)
        self.assertFalse(player.resume().paused)
        self.assertEqual(player.step().position, 1)
        self.assertEqual(player.seek(999).position, len(player.iter_events()) - 1)
        self.assertTrue(player.pause().paused)

    def test_oracle_scheduler_is_offline_plan_not_live_scheduler(self) -> None:
        oracle = OracleScheduler()
        plan = oracle.plan(TraceLoader().from_telemetry(_telemetry()), OracleObjective.BALANCED)

        self.assertGreaterEqual(plan.utility, -100.0)
        self.assertEqual(len(plan.decisions), 3)
        self.assertFalse(hasattr(oracle, "decide"))

    def test_counterfactual_engine_computes_metrics_and_regret(self) -> None:
        trace = TraceLoader().from_telemetry(_telemetry())
        scenario = CounterfactualScenario(
            name="continue_one_more",
            scenario_type=ScenarioType.CONTINUE_EXTRA_LAYERS,
            extra_layers=1,
        )
        result = CounterfactualEngine().evaluate(trace, scenario)

        self.assertEqual(result.scenario.name, "continue_one_more")
        self.assertGreaterEqual(result.metrics.decision_stability, 0.0)
        self.assertGreaterEqual(result.regret.average_regret, 0.0)

    def test_regret_handles_token_sequence_and_normalized_regret(self) -> None:
        regret = compute_regret([1.0, 0.8, 0.6], [0.5, 0.9, 0.2])

        self.assertEqual(len(regret.token_regret), 3)
        self.assertAlmostEqual(regret.sequence_regret, 0.9)
        self.assertGreater(regret.normalized_regret, 0.0)

    def test_batch_regret_differs_from_sequence_regret(self) -> None:
        regret = compute_regret([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])

        self.assertNotEqual(regret.batch_regret, regret.sequence_regret)

    def test_counterfactual_experiment_exports_publication_tables(self) -> None:
        trace = TraceLoader().from_telemetry(_telemetry())
        experiment = CounterfactualExperiment()
        rows = experiment.default_policy_comparison(trace)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = experiment.export(rows, temp_dir)

            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["latex"].exists())
            self.assertTrue(paths["html"].exists())
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["policy"], "static")

    def test_counterfactual_experiment_exports_empty_tables(self) -> None:
        experiment = CounterfactualExperiment()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = experiment.export([], temp_dir)

            self.assertEqual(json.loads(Path(paths["json"]).read_text(encoding="utf-8")), [])
            self.assertIn("policy", Path(paths["csv"]).read_text(encoding="utf-8"))

    def test_counterfactual_experiment_escapes_table_content(self) -> None:
        experiment = CounterfactualExperiment()
        rows = [
            CounterfactualExperiment().default_policy_comparison(
                TraceLoader().from_telemetry(_telemetry())
            )[0]
        ]
        rows[0] = type(rows[0])(
            policy="pvs | <unsafe>",
            utility=rows[0].utility,
            latency_ms=rows[0].latency_ms,
            compute_units=rows[0].compute_units,
            memory_mb=rows[0].memory_mb,
            budget_efficiency=rows[0].budget_efficiency,
            oracle_gap=rows[0].oracle_gap,
            normalized_regret=rows[0].normalized_regret,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = experiment.export(rows, temp_dir)

            self.assertIn("pvs \\| <unsafe>", Path(paths["markdown"]).read_text(encoding="utf-8"))
            self.assertIn("pvs | &lt;unsafe&gt;", Path(paths["html"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
