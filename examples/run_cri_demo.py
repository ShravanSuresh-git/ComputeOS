"""Run a small Counterfactual Runtime Intelligence demo."""

from __future__ import annotations

from computeos.replay import (
    CounterfactualEngine,
    CounterfactualScenario,
    ScenarioType,
    TraceLoader,
    TracePlayer,
)
from computeos.replay.experiment import CounterfactualExperiment, to_markdown
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.telemetry.metrics import ActivationStats, LayerTelemetry, ModelTelemetry


def main() -> None:
    telemetry = _sample_telemetry()
    trace = TraceLoader().from_telemetry(telemetry)
    player = TracePlayer(trace)
    state = player.resume()
    print(f"Replay started at event {state.position}: {state.event.event_type}")
    print(f"Total replay events: {len(player.iter_events())}")

    scenario = CounterfactualScenario(
        name="latency_budget_2ms",
        scenario_type=ScenarioType.CHANGE_LATENCY_BUDGET,
        latency_budget_ms=2.0,
    )
    result = CounterfactualEngine().evaluate(trace, scenario)
    print(f"Scenario: {scenario.name}")
    print(f"Predicted utility: {result.predicted_utility:.6f}")
    print(f"Oracle gap: {result.metrics.oracle_gap:.6f}")
    print(f"Normalized regret: {result.regret.normalized_regret:.6f}")

    rows = CounterfactualExperiment().default_policy_comparison(trace)
    print("\nPolicy comparison")
    print(to_markdown(rows))


def _sample_telemetry() -> ModelTelemetry:
    telemetry = ModelTelemetry(model_name="tiny")
    for index in range(4):
        telemetry.layers.append(
            LayerTelemetry(
                layer_name=f"transformer.h.{index}",
                layer_type="GPT2Block",
                latency_ms=1.0 + index,
                activation_stats=ActivationStats(
                    mean=0.1 * index,
                    std=0.2,
                    min=-0.5,
                    max=0.5,
                    l2_norm=2.0 + index,
                    numel=1024 * (index + 1),
                ),
                process_rss_bytes=(128 + index) * 1024 * 1024,
            )
        )
        telemetry.scheduler_decisions.append(
            SchedulerDecision(
                action=SchedulerAction.EARLY_EXIT if index == 2 else SchedulerAction.CONTINUE,
                layer_name=f"transformer.h.{index}",
                confidence=0.7,
                reason="demo decision",
                metadata={
                    "prediction": {
                        "expected_improvement": 0.4 - index * 0.05,
                        "expected_net_value": 0.2 - index * 0.1,
                    }
                },
            )
        )
    telemetry.confidence_scores.extend([0.6, 0.7, 0.8])
    telemetry.total_latency_ms = 10.0
    telemetry.peak_process_rss_bytes = 132 * 1024 * 1024
    return telemetry


if __name__ == "__main__":
    main()
