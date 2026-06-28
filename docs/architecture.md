# ComputeOS Architecture

ComputeOS separates policy research from model execution. The framework is built
around five boundaries:

1. **Configuration** defines reproducible experiments with Hydra YAML and typed
   dataclasses in `computeos.config`.
2. **Instrumentation** attaches PyTorch hooks to transformer blocks discovered
   from Hugging Face and PyTorch naming conventions. Hooks record observations
   without mutating model parameters or replacing layers.
3. **Telemetry** owns structured metrics: per-layer latency, activation
   statistics, attention entropy, confidence scores, memory usage, and scheduler
   decisions.
4. **Scheduling** exposes one policy API. Heuristics, learned classifiers, and
   reinforcement learning agents all implement `Scheduler.decide`.
5. **Execution and benchmarking** coordinate tokenization, generation,
   instrumentation, scoring, and logging.

## Forward Pass Instrumentation

`HookedTransformerMonitor` registers a forward pre-hook and forward hook on
likely transformer blocks. The pre-hook records the start timestamp. The
post-hook computes telemetry from detached outputs, constructs a
`SchedulerContext`, asks the scheduler for a decision, and records that decision.

This first version is observational by default. Future execution engines can act
on `SchedulerAction.SKIP_LAYER`, `EARLY_EXIT`, or `ADJUST_CACHE` by adding
controlled adapters around model execution while keeping the scheduler API
stable.

## Scheduler Contract

Schedulers implement:

- `reset()` before each benchmark item or prompt.
- `decide(context)` for inference-time decisions.
- `observe(context, decision)` for optional online updates.

The baseline `HeuristicScheduler` demonstrates the contract while leaving room
for research policies:

- A supervised classifier can read `context.layer_telemetry` and emit early-exit
  decisions with calibrated confidence.
- An RL scheduler can override `observe` to update a replay buffer or reward
  trace.
- A systems policy can use memory telemetry to adjust KV-cache strategy.

## Telemetry Model

Telemetry records are dataclasses so they are lightweight, typed, and easy to
serialize. Layer telemetry includes:

- layer name and module type
- latency in milliseconds
- activation mean, standard deviation, min, max, L2 norm, and element count
- attention entropy when attention probabilities are available
- CUDA allocated and reserved memory
- process RSS memory for CPU-only and mixed deployments

Model telemetry includes:

- total latency
- peak memory
- all layer records
- scheduler decisions
- generated-token confidence scores from model logits
- scheduler confidence scores when policies emit them

## Benchmarking and Logging

Benchmarks implement `Benchmark.items()` and optionally `Benchmark.score()`.
The smoke benchmark runs a list of prompts. Standard benchmark integrations
should live behind this interface so execution and telemetry remain reusable.

W&B, JSON, and CSV support are implemented as telemetry loggers. W&B imports are
lazy, JSON export writes model-level records, and CSV export writes one row per
layer event. Tests and offline workflows can use `InMemoryTelemetryLogger`.

Human-readable terminal output is handled by `computeos.telemetry.reports`, which
keeps reporting separate from collection and export.

## Extension Checklist

To add a scheduler:

1. Create `src/computeos/scheduling/<policy>.py`.
2. Subclass `Scheduler`.
3. Register a factory in `default_scheduler_registry`.
4. Add `conf/scheduler/<policy>.yaml`.
5. Add unit tests for decisions and state reset behavior.

To add a benchmark:

1. Subclass `Benchmark`.
2. Register it in `default_benchmark_registry`.
3. Add a Hydra config under `conf/benchmark`.
4. Add scoring tests with deterministic execution fixtures.

To add telemetry:

1. Extend the telemetry dataclasses.
2. Compute the value in `telemetry.stats` or the hook manager.
3. Emit it in W&B logging.
4. Add a unit test that verifies tensors are detached and not retained.
