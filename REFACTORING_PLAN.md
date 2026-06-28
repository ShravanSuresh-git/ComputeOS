# ComputeOS Refactoring Plan

## Migration Philosophy

This plan intentionally avoids a breaking rewrite. Every step should:

- compile successfully
- preserve current public imports
- preserve current examples
- preserve Hydra CLI behavior
- include tests
- include documentation
- be small enough for one reviewable pull request

## Phase 0: Stabilize Current Contracts

### Step 0.1: Add Public API Contract Tests

- Change: add tests for imports from `computeos`, `computeos.scheduling`,
  `computeos.telemetry`, and `computeos.benchmarks`.
- Backwards compatibility: full.
- Tests: import and instantiate public classes.
- Docs: update CONTRIBUTING with public API expectations.

### Step 0.2: Add CLI Smoke Test

- Change: test `computeos-run --help` or a minimal Hydra invocation with an
  offline fake backend once available.
- Backwards compatibility: full.
- Tests: CI CLI smoke test.
- Docs: quickstart verification section.

### Step 0.3: Run Ruff In CI

- Change: add Ruff check to CI.
- Backwards compatibility: full.
- Tests: existing test suite plus lint.
- Docs: no new docs required.

## Phase 1: Introduce Runtime Events Without Behavior Changes

### Step 1.1: Add `runtime/events.py`

- Change: add typed dataclasses for runtime events.
- Backwards compatibility: no existing code removed.
- Tests: serialization and field tests.
- Docs: telemetry lifecycle section.

### Step 1.2: Mirror Current Telemetry Into Events

- Change: `HookedTransformerMonitor` emits events internally while still filling
  `ModelTelemetry`.
- Backwards compatibility: existing telemetry remains unchanged.
- Tests: hook tests assert both telemetry and events.
- Docs: architecture update.

### Step 1.3: Add Trace Store Interface

- Change: introduce `TraceStore` protocol and in-memory implementation.
- Backwards compatibility: existing loggers still work.
- Tests: append/read events.
- Docs: experiment artifact guide.

## Phase 2: Separate Observation From Scheduling

### Step 2.1: Add `ObservationFrame`

- Change: derive scheduler-facing observations from events and current telemetry.
- Backwards compatibility: adapt frame to existing `SchedulerContext`.
- Tests: frame construction from layer telemetry.
- Docs: scheduler guide update.

### Step 2.2: Add `SchedulerRuntime`

- Change: move scheduler invocation out of hook callbacks into a small runtime
  coordinator.
- Backwards compatibility: `HookedTransformerMonitor` still accepts scheduler
  and collector, but delegates internally.
- Tests: scheduler invocation order.
- Docs: runtime lifecycle diagram.

### Step 2.3: Add Decision Validation

- Change: validate scheduler decisions against known action enum and current
  backend capabilities, initially observational only.
- Backwards compatibility: unsupported actions become recorded rejections, not
  crashes.
- Tests: unsupported action result.
- Docs: action result reference.

## Phase 3: Backend Capability Layer

### Step 3.1: Add `BackendCapabilities`

- Change: dataclass describing observation and action capabilities.
- Backwards compatibility: current backend declares observe-only capabilities.
- Tests: capability defaults and feature flags.
- Docs: capability matrix.

### Step 3.2: Add `ExecutionBackend` Protocol

- Change: define backend interface without migrating `InferenceEngine` yet.
- Backwards compatibility: no behavior change.
- Tests: fake backend implements protocol.
- Docs: backend author guide.

### Step 3.3: Wrap Current Engine As Backend Adapter

- Change: create `HuggingFaceGenerateBackend` that delegates to current logic.
- Backwards compatibility: `InferenceEngine` remains available.
- Tests: existing examples pass.
- Docs: migration notes.

## Phase 4: Experiment Artifacts

### Step 4.1: Add `ExperimentStore`

- Change: store resolved config, environment info, traces, metrics, and summary.
- Backwards compatibility: optional path, no behavior change by default.
- Tests: temp-dir artifact creation.
- Docs: quickstart artifact section.

### Step 4.2: Add Metric Aggregators

- Change: aggregate latency, memory, decision counts, action results, and quality.
- Backwards compatibility: existing telemetry exports remain.
- Tests: aggregation from sample telemetry.
- Docs: metrics reference.

### Step 4.3: Add Baseline Comparison

- Change: support running baseline and scheduled policy with same prompts.
- Backwards compatibility: disabled by default.
- Tests: fake engine comparison.
- Docs: benchmark guide.

## Phase 5: Controlled PyTorch Backend

### Step 5.1: Add Minimal Controlled Decode Loop

- Change: implement a backend for a small causal LM family with explicit token
  loop and layer events.
- Backwards compatibility: default backend remains current Hugging Face path.
- Tests: tiny local model decode loop.
- Docs: backend limitations.

### Step 5.2: Implement `EARLY_EXIT` Action For Controlled Backend

- Change: support one real adaptive action.
- Backwards compatibility: unsupported backends record rejected action results.
- Tests: early exit reduces executed layers in fake/tiny backend.
- Docs: first adaptive runtime tutorial.

### Step 5.3: Add Online Confidence Signals

- Change: expose logits confidence during decode, not only after generation.
- Backwards compatibility: post-hoc confidence remains.
- Tests: scheduler receives confidence before action.
- Docs: telemetry signal reference.

## Phase 6: Plugin System

### Step 6.1: Add Entry-Point Discovery

- Change: discover schedulers, benchmarks, backends, and telemetry sinks from
  package entry points.
- Backwards compatibility: manual registries still work.
- Tests: local fake entry point or registry injection.
- Docs: plugin author guide.

### Step 6.2: Add Plugin Metadata

- Change: plugin factories expose name, version, config schema, capabilities,
  and compatibility.
- Backwards compatibility: default metadata for existing components.
- Tests: metadata validation.
- Docs: plugin manifest reference.

### Step 6.3: Add Config Schema Registration

- Change: allow plugins to provide typed config schemas.
- Backwards compatibility: generic `parameters` remains supported.
- Tests: typed config creation.
- Docs: config guide.

## Phase 7: Benchmark Maturity

### Step 7.1: Add Perplexity Scorer

- Change: implement log-likelihood/perplexity benchmark path.
- Backwards compatibility: prompt smoke unchanged.
- Tests: deterministic tiny model scorer.
- Docs: WikiText benchmark guide.

### Step 7.2: Add Quality/Latency Pareto Reports

- Change: generate summaries across thresholds and policies.
- Backwards compatibility: optional.
- Tests: synthetic metrics produce expected frontier.
- Docs: experiment report guide.

### Step 7.3: Add Standard Benchmark Suite Configs

- Change: add configs for common benchmark subsets with optional dependencies.
- Backwards compatibility: optional extras.
- Tests: config load tests only.
- Docs: benchmark matrix.

## Phase 8: First Publishable Scheduler

### Step 8.1: Implement Confidence-Aware Early Exit

- Change: implement first action-applying scheduler for controlled backend.
- Backwards compatibility: scheduler disabled by default.
- Tests: threshold behavior, action results, quality checks on fixtures.
- Docs: algorithm note.

### Step 8.2: Add Evaluation Script

- Change: run baseline versus early-exit over a small benchmark subset.
- Backwards compatibility: optional example.
- Tests: dry-run mode.
- Docs: reproduction instructions.

### Step 8.3: Add Paper-Ready Artifact Template

- Change: generate `summary.md`, metrics JSON, and plots.
- Backwards compatibility: optional.
- Tests: artifact snapshot.
- Docs: publication guide.

## Commit Strategy

Recommended pull request sequence:

1. Public API contract tests.
2. Runtime event dataclasses.
3. Trace store.
4. Observation frame.
5. Scheduler runtime coordinator.
6. Backend capabilities.
7. Backend protocol.
8. Current engine backend adapter.
9. Experiment store.
10. Metric aggregators.
11. Controlled PyTorch backend.
12. First enforceable action.
13. Plugin discovery.
14. Benchmark scoring.
15. First publishable scheduler.

No PR should combine more than one architectural boundary change.

## Rollback Strategy

Each migration step should be revertible without data loss:

- preserve old public imports
- leave old runner path intact until replacement is tested
- add adapters before removing direct calls
- gate new behavior behind config flags
- write compatibility tests before refactoring internals

## Success Criteria

ComputeOS V2 is ready for publishable research when:

- a scheduler can apply at least one real runtime action
- action results are recorded separately from requested decisions
- baseline comparisons are automatic
- benchmark quality metrics exist
- experiment artifacts are reproducible
- plugin schedulers can be added without editing core files
- documentation explains backend capabilities and limitations
