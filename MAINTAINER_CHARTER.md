# ComputeOS Maintainer Charter

## Purpose

ComputeOS exists to become the reference research platform for adaptive AI
inference. It is not an inference engine, model wrapper, benchmark harness, or
agent framework. It is the runtime research layer where new scheduling
algorithms can be designed, observed, benchmarked, and compared.

The long-term ambition is to make ComputeOS feel like infrastructure: a system
researchers can trust for years, extend without rewriting internals, and cite in
papers about adaptive inference.

## Product Thesis

Current inference systems mostly execute static plans. ComputeOS studies runtime
intelligence:

- observe execution continuously
- estimate utility
- allocate compute, memory, precision, and scheduling priority
- optimize under latency, memory, and compute constraints
- log every decision and outcome

The core research question is:

> Can inference runtimes allocate computation dynamically, per request, token,
> layer, and resource, while preserving quality and reproducibility?

## What ComputeOS Is

- A runtime kernel for adaptive inference research.
- A scheduler SDK.
- A telemetry and trace engine.
- A plugin platform.
- An experiment and benchmark substrate.
- A mathematical scheduling framework.
- A foundation for paper-ready evaluation.

## What ComputeOS Is Not

- Not another inference engine.
- Not another LLM wrapper.
- Not another chatbot framework.
- Not another agent framework.
- Not a collection of disconnected heuristics.
- Not a benchmark leaderboard with incidental runtime code.

ComputeOS may integrate with engines such as Hugging Face Transformers, vLLM,
SGLang, and TensorRT-LLM, but it should not duplicate their serving goals.

## Engineering Values

Every change should improve at least one of:

- architecture
- maintainability
- clarity
- extensibility
- research usability
- determinism
- observability
- benchmarkability

Complexity is acceptable only when it creates measurable research or
maintainability value.

## Design Principles

### Use

- SOLID design
- dependency injection
- composition over inheritance
- protocols and small interfaces
- immutable runtime context
- pure scheduler APIs
- plugin-based extension
- event-driven execution
- typed dataclasses
- deterministic tests
- documented tradeoffs

### Avoid

- global mutable state
- singleton-heavy architecture
- hidden dependencies
- magic numbers
- hard-coded scheduler behavior
- duplicate runtime logic
- scheduler code that directly executes inference
- features without tests
- documentation that overstates implemented behavior

## Architectural Contract

ComputeOS must keep these responsibilities separate:

```text
Runtime observes execution.
Scheduler decides what should happen.
Runtime applies supported decisions.
Telemetry records what happened.
Benchmarks evaluate consequences.
Experiments preserve artifacts.
```

Schedulers never directly execute inference. They emit immutable decisions.
Runtime components validate and apply decisions. Every decision and action result
must be inspectable.

## Runtime Principles

The runtime should become event-driven. Every significant boundary should be
traceable:

- request start/end
- decode step start/end
- layer start/end
- logits ready
- token selected
- cache updated
- scheduler decision
- action accepted/rejected/applied
- runtime error

Runtime events should eventually include request IDs, token indices, layer
indices, device/rank metadata, timestamps, backend capabilities, and resource
state.

## Scheduler Principles

Schedulers should be:

- backend-aware through declared capabilities
- deterministic under fixed seeds/configs
- independent from model execution internals
- small enough for researchers to implement quickly
- benchmarkable against static execution and baseline policies
- explicit about required telemetry

Scheduler outputs must be immutable and logged. Scheduler failures should become
structured runtime events where possible.

## Telemetry Principles

Everything should be observable, but not everything should be collected all the
time. Telemetry should support:

- layer timings
- scheduler decisions
- action results
- memory usage
- GPU utilization
- attention statistics
- confidence
- runtime events
- latency
- estimated FLOPs
- resource allocation
- utility estimates
- budget consumption

Telemetry collection must be configurable, sampleable, and measurable in terms
of overhead.

## Benchmarking Principles

Every scheduling algorithm should be compared against:

- static execution
- simple heuristics
- existing adaptive policies
- ablations of its own components

Benchmark artifacts should include:

- resolved config
- JSON metrics
- CSV exports
- trace files
- publication-quality figures
- LaTeX tables
- interactive reports when available

## Mathematical Standard

Scheduling algorithms should include a mathematical formulation when practical:

- objective
- variables
- constraints
- utility function
- cost model
- decision rule
- assumptions
- failure modes

ComputeOS is a research project. Code and math should reinforce each other.

## Implementation Workflow

Before implementing a feature:

1. inspect the existing implementation
2. identify extension points
3. write an implementation plan
4. explain architectural tradeoffs
5. implement incrementally
6. add tests
7. add documentation
8. update diagrams if architecture changes

No large monolithic rewrites. No breaking changes without a migration path.

## Quality Bar

A change is not complete until:

- public behavior is documented
- tests cover the new contract
- telemetry implications are clear
- benchmark implications are clear
- compatibility implications are clear
- unsupported behavior is explicit

## North Star

ComputeOS should let a researcher implement a new adaptive inference scheduling
algorithm in a few hundred lines of code, run it against credible baselines, and
produce reproducible artifacts suitable for a paper.
