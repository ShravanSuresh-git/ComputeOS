# Quickstart

This guide walks through the shortest path from a fresh checkout to useful
ComputeOS telemetry.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 2. Run the Offline Tests

```bash
python -m unittest discover -s tests
```

The tests use tiny local PyTorch modules, so they do not download Hugging Face
models or benchmark datasets.

## 3. Run the Tiny GPT-2 Demo

```bash
python examples/run_tiny_gpt2.py
```

The demo downloads `sshleifer/tiny-gpt2`, runs a short generation, and prints a
terminal telemetry report with latency, scheduler decisions, memory, and
activation summaries.

## 4. Export Telemetry

Use Hydra overrides from the CLI:

```bash
computeos-run \
  model.name=sshleifer/tiny-gpt2 \
  execution.max_new_tokens=8 \
  telemetry.export_path=outputs/tiny-gpt2.json \
  telemetry.export_format=json
```

CSV export writes one row per observed layer event:

```bash
computeos-run telemetry.export_path=outputs/layers.csv telemetry.export_format=csv
```

## 5. Change the Scheduler

Schedulers are selected from config:

```bash
computeos-run scheduler=heuristic scheduler.parameters.entropy_threshold=1.0
```

To add a new scheduler, start with [examples/custom_scheduler.py](../examples/custom_scheduler.py)
and then follow [docs/scheduler_guide.md](scheduler_guide.md).

## 6. Try a Benchmark Adapter

Prompt smoke runs out of the box. WikiText is available as an optional adapter:

```bash
python -m pip install datasets
computeos-run benchmark=wikitext_perplexity benchmark.limit=8
```

The current WikiText adapter focuses on data loading and telemetry capture. A
future scorer should add log-likelihood/perplexity execution without changing
the benchmark registration pattern.
