# Scheduler Guide

ComputeOS schedules inference through one small policy interface:

```python
class Scheduler:
    def reset(self) -> None: ...
    def decide(self, context: SchedulerContext) -> SchedulerDecision: ...
    def observe(self, context: SchedulerContext, decision: SchedulerDecision) -> None: ...
```

The execution engine owns tokenization, generation, hooks, and telemetry. A
scheduler only receives a `SchedulerContext` and returns a `SchedulerDecision`.

## Heuristic Scheduler

Use this for thresholds, guardrails, and interpretable baselines.

Good inputs:

- `context.layer_telemetry.latency_ms`
- `context.layer_telemetry.activation_stats`
- `context.layer_telemetry.attention_entropy`
- `context.model_telemetry.confidence_scores`

Good outputs:

- `SchedulerAction.RECORD_ONLY` for observability baselines
- `SchedulerAction.CONTINUE` with metadata for policy traces
- Future execution adapters can act on `EARLY_EXIT`, `SKIP_LAYER`, or
  `ADJUST_CACHE`

## Learned Classifier Scheduler

A classifier scheduler should keep model probes separate from model weights.
Useful first version:

1. Extract activation statistics from `LayerTelemetry`.
2. Feed those features into a small calibrated classifier.
3. Emit confidence in `SchedulerDecision.confidence`.
4. Record thresholds and calibration metadata in `SchedulerDecision.metadata`.

## Predictive Value Scheduling

Predictive Value Scheduling is the first research-grade scheduler design in
ComputeOS. It treats adaptive inference as an optimal stopping problem and emits
decisions based on expected marginal improvement minus expected resource cost.

Use it from Hydra:

```bash
computeos-run scheduler=pvs
```

See [predictive_value_scheduling.md](predictive_value_scheduling.md) for the
mathematical formulation, runtime architecture, telemetry, replay support,
benchmark plan, and visualization workflow.

## Reinforcement Learning Scheduler

An RL scheduler can override `observe` to track rewards and online state.
Keep these pieces isolated:

- policy state
- reward shaping
- replay buffer or trajectory storage
- execution action mapping

The scheduler API should stay stable even if the trainer becomes more complex.

## Registration Checklist

1. Add a file under `src/computeos/scheduling/`.
2. Subclass `Scheduler`.
3. Register a factory in `default_scheduler_registry`.
4. Add a config under `conf/scheduler/`.
5. Add tests for `reset`, `decide`, and any threshold behavior.
6. Add an experiment note or example command.
