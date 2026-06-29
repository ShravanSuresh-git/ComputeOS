# Changelog

## 1.1.0 — 2026-06-28

### Added

- Token-level manual decode loop in `InferenceEngine` — `EARLY_EXIT` is now
  enforced at each token step, not just recorded.
- `BackendCapabilities` dataclass — backends declare which actions they support;
  schedulers receive this in `SchedulerContext` and degrade gracefully.
- `PerplexityBenchmark` — per-token perplexity scoring from live telemetry.
- Baseline schedulers — `EntropyScheduler`, `ConfidenceScheduler`,
  `RandomScheduler` — all registered and contract-tested.
- `PolicyComparisonRunner` and `ComparisonReport` — multi-scheduler comparison
  harness producing CSV, JSON regret, and Pareto frontier.
- `ArtifactStore` — reproducibility snapshots of config, environment, telemetry,
  and comparison reports for every run.
- `calibrate_weights()` — data-driven PVS weight calibration via
  `scipy.optimize.nnls` over oracle utilities from real traces.
- `ParetoAnalyzer` — quality/latency Pareto frontier with optional
  `matplotlib` plot output.
- `ControlledBenchmarkRunner` — adapter connecting `ControlledForwardRuntime`
  (which enforces `SKIP_LAYER` and layer budgets) to the benchmark path.
- `warmup_runs` and `n_runs` fields in `ExecutionConfig` — GPU warm-up and
  repeated-run variance reporting.
- `request_id` in `ModelTelemetry` — every inference trace is uniquely
  identified across runs.
- Online confidence scores — `collector.push_confidence_score()` makes scores
  available to schedulers during generation, not only after.
- `attention_entropy_available` flag on `LayerTelemetry` — explicit signal
  when entropy could not be extracted from the hook output.
- `sample_rate` honored in `HookedTransformerMonitor` — probabilistic layer
  sampling for reduced overhead on large models.
- CI gates — ruff lint, mypy strict, and scheduler contract tests are all
  required on every push.

### Fixed

- `batch_regret` was a copy of `sequence_regret`. It is now the mean per-token
  regret, distinct from the sequence total.
- `snapshot_report()` was defined but never called. It is now called from both
  `PolicyComparisonRunner.run()` and `_snapshot_artifacts()` in the runner.
- `ControlledForwardRuntime` now exposes a `capabilities` property and passes
  it into every `SchedulerContext`.
- `compare_schedulers.py` example now uses a single shared engine with
  `PolicyComparisonRunner`'s scheduler-swap mechanism.
- `calibrate_weights()` now raises a clear `ImportError` with install
  instructions when `scipy` is absent.
- mypy strict type error on `engine.py:169` (`str | list[str]` vs `str`) fixed
  with an explicit `str()` cast.

### Notes

- `ADJUST_CACHE` remains in the action enum but is not yet enforced by any
  backend. Tracked in `TECHNICAL_DEBT.md`.
- Standard downstream task evals (MMLU, HellaSwag) are not yet integrated.
  Tracked in `TECHNICAL_DEBT.md`.
- `paper/paper.md` experiment sections remain as structured placeholders.
  Fill them after running `examples/compare_schedulers.py` with a real model.
