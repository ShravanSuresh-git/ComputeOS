# ComputeOS Architecture Review

## Scope

This review covers the repository at commit `0e3df62` and inspects all source,
configuration, documentation, examples, tests, packaging metadata, CI metadata,
and GitHub issue templates. The goal is to evaluate ComputeOS as a future
research platform for adaptive inference scheduling, not as a one-off demo.

## Executive Summary

ComputeOS is currently a clean, small, well-separated research scaffold. Its
main strengths are explicit module boundaries, hook-based non-invasive
instrumentation, typed dataclass configuration, deterministic tests, and clear
extension points for schedulers, telemetry sinks, and benchmark adapters.

The most important architectural limitation is that the runtime is observational.
Schedulers can emit actions such as `EARLY_EXIT`, `SKIP_LAYER`, and
`ADJUST_CACHE`, but the execution engine does not yet enforce those actions.
ComputeOS therefore measures adaptive scheduling conditions, but it is not yet
an adaptive runtime.

The second major limitation is that the framework is coupled to Hugging Face
`generate` as a black-box lifecycle. That is a reasonable first backend, but it
does not expose enough control over token-level, layer-level, cache-level, or
batch-level execution to support the more ambitious research agenda.

The right evolution path is not a rewrite. The existing abstractions should be
kept and surrounded by missing runtime interfaces: execution backends,
observation buses, action applicators, scheduler lifecycle hooks, experiment
artifacts, benchmark scorers, and plugin discovery.

## Repository Inventory

### Packaging and Project Metadata

- `pyproject.toml`
  - Python target: `>=3.12,<3.13`
  - Core dependencies: PyTorch, Transformers, Hydra, OmegaConf, W&B, NumPy,
    psutil, Pydantic, Rich
  - Optional benchmark dependency: `datasets`
  - Tooling: Hatchling, Ruff, mypy, pytest metadata
- `LICENSE`
  - MIT license
- `.gitignore`
  - Python caches, virtualenvs, W&B, Hydra outputs, `.DS_Store`

### Configuration

- `src/computeos/config/schema.py`
  - Dataclass configs for model, scheduler, telemetry, execution, benchmark,
    and top-level experiment
- `conf/`
  - Hydra defaults for model, scheduler, telemetry, execution, and benchmarks

### Scheduling

- `src/computeos/scheduling/base.py`
  - Abstract `Scheduler` with `reset`, `decide`, and optional `observe`
- `src/computeos/scheduling/context.py`
  - Immutable `SchedulerContext`
- `src/computeos/scheduling/decision.py`
  - `SchedulerAction` enum and `SchedulerDecision`
- `src/computeos/scheduling/heuristic.py`
  - Baseline observational heuristic
- `src/computeos/scheduling/registry.py`
  - Explicit registry and factory for schedulers

### Execution

- `src/computeos/execution/model_loader.py`
  - Hugging Face causal LM loading
- `src/computeos/execution/engine.py`
  - Tokenization, `model.generate`, hook registration, telemetry finalization,
    and token confidence extraction

### Instrumentation

- `src/computeos/instrumentation/layers.py`
  - Transformer block discovery by class names and common module name patterns
- `src/computeos/instrumentation/hooks.py`
  - Forward pre-hook/post-hook monitor that records telemetry and calls scheduler

### Telemetry

- `src/computeos/telemetry/metrics.py`
  - Dataclasses for activation, layer, and model telemetry
- `src/computeos/telemetry/stats.py`
  - Tensor traversal, activation stats, attention entropy heuristics
- `src/computeos/telemetry/collector.py`
  - Mutable per-request telemetry collection
- `src/computeos/telemetry/loggers.py`
  - In-memory, composite, JSON, CSV, and W&B loggers
- `src/computeos/telemetry/serialization.py`
  - JSON and CSV flattening helpers
- `src/computeos/telemetry/reports.py`
  - Rich/plain terminal report rendering

### Benchmarks

- `src/computeos/benchmarks/base.py`
  - `Benchmark`, `BenchmarkItem`, and `BenchmarkResult`
- `src/computeos/benchmarks/prompt_smoke.py`
  - Deterministic prompt list benchmark
- `src/computeos/benchmarks/wikitext.py`
  - Optional WikiText data adapter
- `src/computeos/benchmarks/registry.py`
  - Explicit benchmark registry

### Experiments

- `src/computeos/experiments/runner.py`
  - Hydra CLI, model loading, registry construction, benchmark run, logging,
    optional terminal report

### Tests

- `tests/test_scheduler.py`
- `tests/test_hooks.py`
- `tests/test_telemetry_stats.py`
- `tests/test_telemetry_outputs.py`
- `tests/test_benchmarks.py`

The tests are offline, small, and deterministic. They verify contracts rather
than model quality.

### Documentation and Examples

- `README.md`
- `CONTRIBUTING.md`
- `docs/architecture.md`
- `docs/quickstart.md`
- `docs/scheduler_guide.md`
- `docs/research_roadmap.md`
- `examples/run_tiny_gpt2.py`
- `examples/custom_scheduler.py`
- `.github/ISSUE_TEMPLATE/confidence_early_exit.md`

## Current Strengths

### Clear Separation of Concerns

The package layout is easy to reason about:

- Scheduling is policy-facing.
- Execution owns runtime orchestration.
- Instrumentation owns hooks.
- Telemetry owns records, summaries, and sinks.
- Benchmarks own data/scoring structure.
- Config owns typed experiment parameters.

This is the right foundation for a research framework. New functionality can be
placed without guessing where it belongs.

### Non-Invasive Instrumentation

The hook design honors the core ComputeOS philosophy: observe model execution
without modifying weights. That is important for reproducibility, comparison
against unmodified baselines, and compatibility with Hugging Face models.

### Explicit Scheduler API

The scheduler contract is intentionally small:

- `reset`
- `decide`
- `observe`

This is a good starting point because it keeps heuristics, classifiers, and RL
agents behind one interface.

### Typed Telemetry Records

Telemetry dataclasses make logs structured and inspectable. The current metrics
capture the beginnings of what adaptive scheduling needs: latency, activations,
confidence, memory, attention entropy, and decisions.

### Deterministic Offline Tests

Tests do not require model downloads, GPUs, W&B credentials, or benchmark data.
That is exactly the right default for a research framework that should remain
easy to contribute to.

### Lazy Optional Dependencies

The WikiText adapter imports `datasets` only when used. W&B is imported lazily.
This keeps import-time behavior lightweight.

### Lightweight Examples

The tiny GPT-2 example demonstrates an end-to-end run without requiring a large
model. The custom scheduler example helps new researchers understand the policy
contract.

## Architectural Weaknesses

### Scheduler Actions Are Not Applied

`SchedulerAction` includes `EARLY_EXIT`, `SKIP_LAYER`, and `ADJUST_CACHE`, but
the runtime records decisions only. There is no action executor, enforcement
contract, state transition, rollback path, or correctness guardrail.

Impact:

- Researchers may believe policies can alter runtime behavior when they cannot.
- Benchmark results cannot yet measure true compute savings from scheduling.
- The current heuristic is observability, not adaptive inference.

Required abstraction:

- `ActionApplicator` or `RuntimeController`
- capability negotiation between scheduler and backend
- action validation and fallback behavior

### Execution Is Coupled to Hugging Face `generate`

`InferenceEngine.generate` delegates generation to `model.generate`. This is
convenient, but it hides the token loop and limits runtime control.

Limitations:

- No pre-token scheduling boundary.
- No post-logit scheduling boundary before token selection.
- No direct KV-cache management.
- No dynamic batch scheduling.
- No layer skipping enforcement.
- No ability to compare decoding strategies.

Required abstraction:

- `ExecutionBackend`
- `GenerationLoop`
- `DecodeStep`
- `LayerBoundary`
- `CacheState`

### Hooks Are Both Instrumentation and Scheduling Boundary

`HookedTransformerMonitor` records latency, computes stats, constructs
`SchedulerContext`, calls `decide`, records decisions, and calls `observe`.

This is too much responsibility for one class. Hooks should emit observations;
schedulers should be invoked by a runtime loop or observation dispatcher.

Impact:

- Hard to test scheduling separately from PyTorch hook mechanics.
- Hard to support non-hook backends.
- Hard to introduce asynchronous telemetry or batched policy evaluation.

Required abstraction:

- `ObservationBus`
- `TelemetryProbe`
- `SchedulerRuntime`

### Bidirectional Coupling Between Telemetry and Scheduling

`ModelTelemetry` imports `SchedulerDecision`, and `TelemetryCollector` records
scheduler decisions. Meanwhile `SchedulerContext` imports telemetry types.

This is understandable for a small repo, but as the framework grows it creates
a cycle between policy and observability.

Better direction:

- Keep decisions as experiment events in an event log.
- Let telemetry metrics remain policy-neutral.
- Represent scheduler decisions as a typed event stream rather than a field on
  model telemetry.

### Registries Are Manual and Closed

Scheduler and benchmark registries are explicit in-code registries. This is good
for initial clarity but not enough for a plugin ecosystem.

Limitations:

- Third-party schedulers require editing core code.
- No entry-point discovery.
- No metadata for capabilities, required telemetry, or backend compatibility.

Required abstraction:

- plugin entry points
- scheduler metadata
- benchmark metadata
- backend capability declarations

### Configuration Uses Generic Parameter Dictionaries

`SchedulerConfig.parameters` and `BenchmarkConfig.parameters` are flexible, but
they trade away type safety and validation.

Risk:

- Misconfigured experiments fail late.
- Experiment reproducibility suffers because schemas are implicit.
- Tools cannot infer available parameters.

Better direction:

- typed plugin configs
- Pydantic or Hydra structured config validation
- config versioning and resolved config artifacts

### Telemetry Sampling Is Declared But Not Implemented

`TelemetryConfig.sample_rate` exists but hooks record every event. This is a
small but important mismatch between API and behavior.

Impact:

- Users may expect sampling to reduce overhead.
- Large model runs may generate excessive telemetry.

Required abstraction:

- `TelemetrySampler`
- deterministic sampling policy
- per-signal sampling controls

### Attention Entropy Is Opportunistic

`attention_entropy` attempts to infer attention tensors from hook outputs. For
many Hugging Face blocks, attention probabilities are not present at the layer
hook boundary even when `output_attentions=True` is passed to `generate`.

Impact:

- Entropy is often unavailable.
- Results may differ across architectures.

Better direction:

- architecture-specific probes
- optional model output adapters
- explicit telemetry availability metadata

### Confidence Scores Are Post-Hoc

Token confidence is computed after `generate` returns from output scores. It is
not available to schedulers during the actual decode step.

Impact:

- Confidence-aware scheduling cannot use the current confidence signal online.

Required abstraction:

- token-step lifecycle with logits observation before next step

### Benchmarking Does Not Yet Score Adaptive Quality

`Benchmark.score` defaults to `None`. WikiText loads samples but does not compute
perplexity. The framework lacks standard quality/latency/memory tradeoff metrics.

Required abstraction:

- benchmark scorers
- metric aggregators
- baseline comparisons
- repeated-run variance reporting

## Unnecessary Complexity

The repository is mostly lean. Current complexity is justified by clarity rather
than overengineering. A few points should be watched:

- `TelemetryConfig.enabled` exists but is not honored by the runtime.
- `sample_rate` exists but is not honored.
- `SchedulerAction` advertises actions that are not executable yet.
- JSON and CSV loggers buffer all records until `close`; this is simple but not
  suitable for long experiments.
- Fallback plain reports are useful locally, but the package already declares
  Rich as a dependency. The fallback is defensive rather than essential.

## Coupling Analysis

### Current Dependency Direction

```text
experiments
  -> config
  -> model_loader
  -> execution
  -> registries
  -> telemetry loggers/reports

execution
  -> instrumentation
  -> scheduling
  -> telemetry
  -> transformers/torch

instrumentation
  -> scheduling
  -> telemetry
  -> torch/psutil

scheduling
  -> telemetry

telemetry
  -> scheduling decision
```

The most concerning cycle is:

```text
scheduling -> telemetry -> scheduling
```

This is acceptable in a prototype but should be broken before the project grows
plugin APIs.

### Recommended Direction

```text
runtime_core
  -> events
  -> interfaces

scheduling
  -> runtime_core.interfaces
  -> runtime_core.events

telemetry
  -> runtime_core.events

execution_backends
  -> runtime_core
  -> instrumentation

experiments
  -> registries
  -> runtime_core
```

## Scalability Concerns

### Performance Overhead

Per-layer hooks currently compute activation stats synchronously on every event.
For large models, `mean`, `std`, `min`, `max`, and L2 norm over full activations
will add measurable overhead and may force GPU synchronization.

Mitigations:

- async telemetry collection
- sampled tensor stats
- configurable per-signal capture
- no-sync CUDA timing via events
- warmup and benchmark phases

### Memory Growth

Telemetry is stored in lists for the full request and loggers buffer records
until close. Long benchmarks will accumulate large Python objects.

Mitigations:

- streaming telemetry sinks
- bounded buffers
- per-run flushing
- aggregate-only mode

### Multi-GPU and Distributed Runtime

No rank, device, stream, or process metadata exists today. Tensor parallel,
pipeline parallel, and distributed serving research will require explicit
distributed telemetry schemas.

### Backend Scalability

The current backend is Hugging Face-only. Production inference research will
eventually need optional integrations with vLLM, SGLang, TensorRT-LLM, or custom
PyTorch decode loops.

## Research Limitations

ComputeOS currently cannot evaluate the central research question: whether
adaptive runtime scheduling reduces compute while preserving quality. It can
observe signals, but it cannot yet:

- skip a layer
- exit early
- alter cache policy
- change precision
- route tokens to compute budgets
- schedule across batches
- enforce memory-aware policies
- produce quality/latency Pareto curves
- compare against optimized inference engines

## Missing Abstractions

### Runtime Interfaces

- `ExecutionBackend`
- `RuntimeSession`
- `DecodeStep`
- `LayerBoundary`
- `RuntimeCapability`
- `ActionApplicator`
- `ActionResult`

### Scheduler Lifecycle

- `configure`
- `on_request_start`
- `on_decode_step_start`
- `on_layer_start`
- `on_layer_end`
- `on_logits`
- `on_request_end`

### Telemetry Lifecycle

- event schema
- sampler
- async collector
- signal registry
- aggregation layer
- trace IDs
- run IDs
- backend metadata

### Plugin Architecture

- entry-point based discovery
- plugin manifests
- capability declarations
- config schema registration
- version compatibility

### Experiment Lifecycle

- resolved config snapshots
- artifact directories
- run IDs
- baseline comparison plans
- aggregation summaries
- repeat seeds

### Benchmark Lifecycle

- dataset preparation
- prompt formatting
- scoring
- metric aggregation
- quality/latency tradeoff reports

## Performance Bottlenecks

1. Activation stats over full tensors can synchronize GPU execution.
2. Python hooks add overhead at every transformer block.
3. Token confidence is computed after generation and requires retaining scores.
4. JSON/CSV loggers buffer data in memory.
5. `psutil.Process().memory_info()` per layer can become non-trivial overhead.
6. Rich/plain reporting is fine for demos but should not be part of hot paths.
7. `model.generate` prevents optimized control of decode loop internals.

## Future Extension Points

The best existing extension points are:

- `Scheduler` for policy algorithms
- `SchedulerRegistry` for policy construction
- `TelemetryLogger` for sinks
- `Benchmark` for benchmark adapters
- Hydra configs for experiment composition
- `HookedTransformerMonitor` for observation

The next extension points should be:

- `ExecutionBackend`
- `RuntimeEvent`
- `RuntimeAction`
- `ActionApplicator`
- `TelemetryProbe`
- `TelemetrySampler`
- `ExperimentStore`
- `PluginRegistry`

## File-Level Notes

### `src/computeos/execution/engine.py`

Strengths:

- clear orchestration
- inference mode
- seed control
- telemetry finalization

Concerns:

- coupled to Hugging Face `generate`
- scheduler reset happens per prompt only
- no action application
- token confidence is post-hoc
- no batch support despite `BenchmarkConfig.batch_size`

### `src/computeos/instrumentation/hooks.py`

Strengths:

- non-invasive hooks
- handles cleanup
- clear context manager

Concerns:

- too many responsibilities
- no telemetry sampling
- scheduler invocation belongs in runtime rather than hook callback
- no exception event on scheduler failure
- no per-device timing abstraction

### `src/computeos/scheduling/base.py`

Strengths:

- small and understandable
- supports online learners via `observe`

Concerns:

- lacks capability negotiation
- lacks explicit state lifecycle
- lacks async/batched decision path
- no decision validation contract

### `src/computeos/telemetry/metrics.py`

Strengths:

- simple typed records
- useful initial metrics

Concerns:

- policy decisions stored inside model telemetry couples concerns
- no event IDs or timestamps per layer
- no request ID, token index, batch index, rank, or device metadata

### `src/computeos/benchmarks/base.py`

Strengths:

- minimal benchmark contract

Concerns:

- no aggregation
- no repeat/seeding model
- no baseline comparison
- no task-specific scoring lifecycle

### `src/computeos/experiments/runner.py`

Strengths:

- simple CLI path
- lazy Hydra import
- explicit construction

Concerns:

- hardcodes default registries
- not dependency-injected
- no artifact store
- no structured return type for experiment results

## Overall Assessment

ComputeOS is a good foundation, not yet a publishable adaptive inference
research platform. Its design philosophy is sound: non-invasive observation,
typed telemetry, modular policies, Hydra configuration, and offline tests. The
next phase should preserve those strengths while introducing a real runtime
control plane.

The most important strategic move is to separate three concepts that are
currently blended:

1. observing model execution
2. deciding what should happen
3. applying runtime actions

Once those are separated, ComputeOS can support genuinely novel scheduling
research without forcing every policy to understand PyTorch hooks or Hugging
Face internals.
