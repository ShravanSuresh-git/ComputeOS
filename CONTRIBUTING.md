# Contributing

Thanks for improving ComputeOS. The project is designed for research velocity
with production habits.

Before proposing architectural or runtime changes, read
[MAINTAINER_CHARTER.md](MAINTAINER_CHARTER.md). It defines the project vision,
non-goals, subsystem boundaries, and review bar.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Checks

```bash
python -m unittest discover -s tests
ruff check src tests examples
```

If you have not installed the package in editable mode yet, use
`PYTHONPATH=src python -m unittest discover -s tests`.

Default tests should stay offline and deterministic. Do not require benchmark
downloads, GPU hardware, W&B credentials, or Hugging Face authentication in CI.

## Adding a Scheduler

- Keep policy code under `src/computeos/scheduling/`.
- Do not mutate model weights in scheduler code.
- Do not directly execute inference from scheduler code.
- Return explicit `SchedulerDecision` objects with useful reasons and metadata.
- Document required telemetry and backend capabilities.
- Add a Hydra config under `conf/scheduler/`.
- Add focused unit tests for the policy.

## Adding a Benchmark

- Implement `Benchmark.items`.
- Override `Benchmark.score` only when scoring is deterministic and documented.
- Keep optional dataset dependencies lazy.
- Add a config under `conf/benchmark/`.

## Adding Telemetry

- Extend telemetry dataclasses first.
- Compute summaries from detached tensors only.
- Add JSON/CSV and W&B output coverage where relevant.
- Avoid retaining full activations unless an experiment explicitly owns that
  memory cost.
