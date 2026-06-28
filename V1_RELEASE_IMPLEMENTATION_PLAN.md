# ComputeOS v1.0 Release Implementation Plan

## Audit Summary

The repository is clean and already contains:

- scheduler interfaces and PVS
- hook-based telemetry
- Hugging Face generation runtime
- replay/CRI subsystem
- benchmark foundations
- JSON/CSV/W&B telemetry exports
- docs, examples, CI, and tests

The main release blocker is runtime enforcement. The current Hugging Face
`generate` path records scheduler decisions but cannot safely apply arbitrary
layer actions because `generate` owns the decode loop and model internals.

## Implementation Strategy

Do not rewrite the Hugging Face runtime. Keep it as the compatibility and smoke
runtime. Add a controlled runtime for models whose layer order can be explicitly
executed. This gives ComputeOS a real action-applying runtime while preserving
existing public APIs.

## Planned Changes

### 1. Runtime Enforcement

Add an additive controlled execution runtime:

- `computeos.execution.controlled`
- explicit layer iteration
- pre-layer scheduler decision point
- post-layer telemetry decision point
- `EARLY_EXIT` enforcement
- `SKIP_LAYER` enforcement
- latency/compute/memory budget enforcement
- action result telemetry in decision metadata

The controlled runtime will support research/test modules and future model
adapters. The existing Hugging Face runtime remains available and documented as
observational unless a backend supports action application.

### 2. Benchmark and Report Exports

Add publication-style benchmark report helpers:

- CSV
- JSON
- Markdown
- LaTeX
- HTML

Use existing benchmark and replay data. Do not fabricate scores.

### 3. Metrics Integration

Ensure the existing CRI metrics are exposed through a stable API and benchmark
report rows.

### 4. Replay Validation

Add tests for deterministic pause, resume, step, seek, oracle comparison, and
counterfactual replay.

### 5. Demonstration

Add a lightweight `demo.py` that runs a deterministic local demo without large
dependencies:

- controlled runtime
- scheduler decisions
- replay
- counterfactual analysis
- report export

### 6. Visual Documentation

Add Mermaid diagrams for:

- overall architecture
- runtime lifecycle
- scheduler lifecycle
- telemetry pipeline
- replay system
- PVS
- CRI
- oracle scheduler

### 7. Documentation and Paper

Add or improve:

- README release positioning
- installation/quickstart notes
- runtime enforcement documentation
- benchmark reporting documentation
- paper template under `paper/`

### 8. GitHub Polish

Add:

- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `CITATION.cff`
- release checklist
- project roadmap issue template

### 9. Testing

Add unit/integration tests for:

- controlled runtime action enforcement
- budget enforcement
- benchmark report exports
- release demo import path

## Explicit Limitations

- Hugging Face `generate` remains observational for arbitrary layer actions.
- Controlled runtime is the v1.0 action-applying research runtime.
- True FLOPs/Joules remain estimated unless hardware counters are integrated.
- CRI counterfactuals remain estimates, not measured unobserved outcomes.

## Done Criteria

- all tests pass
- controlled runtime proves decisions can affect execution
- docs explain supported/unsupported runtime actions
- release metadata exists
- final release report summarizes changes and limitations
