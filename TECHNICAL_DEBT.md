# ComputeOS Technical Debt

## Critical

### Scheduler Actions Are Not Enforced

- Category: runtime correctness
- Issue: `EARLY_EXIT`, `SKIP_LAYER`, and `ADJUST_CACHE` can be emitted but are
  only recorded.
- Risk: research results could confuse requested actions with applied actions.
- Effort: 2-4 weeks for an action result layer and one enforceable action.

### No Backend Capability Model

- Category: runtime architecture
- Issue: schedulers cannot know what the backend supports.
- Risk: policies will become backend-specific and brittle.
- Effort: 1 week for metadata, 2-3 weeks with validation and tests.

### Hugging Face `generate` Is a Black Box

- Category: execution control
- Issue: the current engine cannot control the decode loop or apply token/layer
  actions.
- Risk: ComputeOS remains observational.
- Effort: 3-6 weeks for a controlled PyTorch backend for one model family.

### No Benchmark Quality Scoring

- Category: research validity
- Issue: benchmarks run prompts but do not score adaptive quality.
- Risk: no credible latency/quality claims.
- Effort: 1-2 weeks for perplexity scoring, 3-4 weeks for multi-task scoring.

## High

### Hooks Combine Observation and Scheduling

- Category: separation of concerns
- Issue: hook callbacks compute telemetry and invoke schedulers directly.
- Risk: difficult to support non-hook backends or async telemetry.
- Effort: 1-2 weeks for event bus extraction.

### Telemetry and Scheduling Are Cyclically Coupled

- Category: dependency design
- Issue: telemetry stores `SchedulerDecision`; scheduling context imports
  telemetry.
- Risk: plugin APIs become hard to stabilize.
- Effort: 1 week for event types and compatibility adapters.

### `TelemetryConfig.enabled` and `sample_rate` Are Not Honored

- Category: API correctness
- Issue: config advertises controls that do not affect runtime.
- Risk: user confusion and avoidable overhead.
- Effort: 1-2 days for basic behavior, 1 week for robust sampling.

### Token Confidence Is Post-Hoc

- Category: online scheduling
- Issue: token confidence is computed after generation, not during scheduling.
- Risk: confidence-aware policies cannot operate online.
- Effort: 2-4 weeks depending on backend control.

### No Experiment Artifact Store

- Category: reproducibility
- Issue: runs do not snapshot resolved config, environment, metrics, and traces.
- Risk: poor reproducibility for papers.
- Effort: 1-2 weeks.

### Manual Registries Block Third-Party Plugins

- Category: extensibility
- Issue: new schedulers and benchmarks require editing core registry files.
- Risk: plugin ecosystem cannot emerge.
- Effort: 1-2 weeks for entry-point discovery.

## Medium

### JSON and CSV Loggers Buffer In Memory

- Category: scalability
- Issue: exporters keep records until `close`.
- Risk: long benchmark runs use excessive memory.
- Effort: 2-3 days for streaming JSONL/CSV.

### Attention Entropy Availability Is Unclear

- Category: telemetry semantics
- Issue: entropy is inferred opportunistically from outputs.
- Risk: missing values are hard to interpret across models.
- Effort: 1 week for telemetry availability metadata and model-specific probes.

### `BenchmarkConfig.batch_size` Is Unused

- Category: API correctness
- Issue: config declares batch size but benchmark runner is sequential.
- Risk: misleading benchmark configuration.
- Effort: 2-4 days for explicit unsupported warning, 2-3 weeks for batching.

### No Device/Rank Metadata

- Category: distributed readiness
- Issue: telemetry lacks device IDs, stream IDs, ranks, and process group info.
- Risk: difficult to extend to distributed inference.
- Effort: 1 week for schema additions.

### No Structured Error Events

- Category: observability
- Issue: scheduler/backend failures are normal exceptions, not trace events.
- Risk: failed experiments are harder to diagnose.
- Effort: 2-4 days.

### Mypy Strict Mode Is Configured But Not Run In CI

- Category: quality
- Issue: `pyproject.toml` configures strict mypy, CI only runs tests.
- Risk: type regressions go unnoticed.
- Effort: 1 day after resolving type issues.

### Ruff Is Documented But Not Run In CI

- Category: quality
- Issue: README and CONTRIBUTING mention Ruff, CI does not run it.
- Risk: style drift.
- Effort: less than 1 day.

## Low

### Package Metadata Uses Inline License Text

- Category: packaging
- Issue: `pyproject.toml` uses `license = { text = "MIT" }` despite a license
  file existing.
- Risk: minor packaging inconsistency.
- Effort: less than 1 hour.

### README Is Still Introductory

- Category: documentation
- Issue: README does not yet show architecture diagrams or a policy lifecycle.
- Risk: new contributors need to jump into docs quickly.
- Effort: 1 day.

### Model Loader Supports Only Causal LM

- Category: model coverage
- Issue: encoder-decoder and sequence classification models are not supported.
- Risk: benchmark scope is narrow.
- Effort: 1-3 weeks depending on model class.

### Report Formatting Has Plain/Rich Split

- Category: polish
- Issue: fallback reporting is useful but duplicates formatting behavior.
- Risk: minor maintenance overhead.
- Effort: less than 1 day if simplified.

### Example Scheduler Is Not Registered

- Category: developer experience
- Issue: example teaches policy code but not complete plugin registration.
- Risk: users still need to inspect registry internals.
- Effort: 1 day for full example.

## Cross-Cutting Improvements

### Documentation

- Add lifecycle diagrams to README.
- Add capability matrix documentation.
- Add benchmark scoring guide.
- Add telemetry schema reference.

### Testing

- Add contract tests for schedulers.
- Add fake backend tests.
- Add trace serialization golden tests.
- Add config validation tests.
- Add CLI smoke tests.

### Performance

- Add benchmark overhead tests.
- Add telemetry sampling.
- Add CUDA event timing.
- Add streaming exporters.

### Research

- Add baseline comparison harness.
- Add repeat-run variance reporting.
- Add Pareto frontier plotting.
- Add policy artifact logging.
