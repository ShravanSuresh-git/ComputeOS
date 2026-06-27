# ComputeOS

ComputeOS is a production-quality research framework for dynamic inference compute
scheduling in transformer models. It is not a chatbot or end-user application. The
goal is to make it easy to instrument transformer execution, collect detailed
telemetry, and experiment with scheduling policies that allocate inference compute
dynamically.

## Design Principles

- **Non-invasive instrumentation:** model weights and module definitions are not
  modified. ComputeOS observes transformer execution with PyTorch hooks.
- **Scheduler polymorphism:** heuristic, learned classifier, and reinforcement
  learning policies share one interface.
- **Separation of concerns:** scheduling, execution, telemetry, benchmarking, and
  configuration are independent modules.
- **Research velocity with production habits:** type hints, tests, Hydra
  configuration, W&B logging, and explicit extension points are part of the base
  skeleton.

## Repository Layout

```text
src/computeos/
  benchmarks/      Benchmark task interfaces and toy smoke benchmark.
  config/          Dataclass configuration schemas.
  execution/       Inference pipeline and model loading.
  instrumentation/ PyTorch hook management and transformer layer discovery.
  scheduling/      Scheduler protocol, decisions, context, and baseline policy.
  telemetry/       Metrics models, collectors, activation/attention utilities, loggers.
  experiments/     Hydra entrypoint for benchmark execution.
conf/              Hydra YAML defaults.
tests/             Unit tests for interfaces and instrumentation behavior.
docs/              Architecture notes and extension guide.
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
computeos-run model.name=distilgpt2 benchmark.prompts='["The future of systems is"]'
```

For offline smoke tests, the unit tests use tiny local PyTorch modules and do not
download Hugging Face models.

## Extension Points

1. Add a scheduler by subclassing `computeos.scheduling.base.Scheduler`.
2. Register it in `computeos.scheduling.registry.SchedulerRegistry`.
3. Add a Hydra config under `conf/scheduler/`.
4. Implement benchmark-specific prompting/scoring by extending
   `computeos.benchmarks.base.Benchmark`.
5. Add telemetry sinks by implementing `computeos.telemetry.loggers.TelemetryLogger`.

See [docs/architecture.md](docs/architecture.md) for the full architecture.
