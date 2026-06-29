# ComputeOS: A Research Framework for Adaptive Runtime Inference

## Abstract

ComputeOS is a research framework for adaptive runtime scheduling in large
language model inference. It provides scheduler interfaces, telemetry, controlled
runtime enforcement, Predictive Value Scheduling, Counterfactual Runtime
Intelligence, and reproducible benchmark/reporting tools. This paper template
describes the system and leaves experiment sections as placeholders until
measured results are available.

## 1. Introduction

Modern inference engines typically execute static computation plans. ComputeOS
explores whether runtime intelligence can allocate compute adaptively while
preserving quality and reproducibility.

## 2. Motivation

Adaptive inference requires a framework that separates observation, scheduling,
runtime action application, and evaluation. ComputeOS provides that separation.

## 3. Related Work

Placeholder for related work covering adaptive computation, early exit,
speculative decoding, vLLM, TensorRT-LLM, SGLang, Hugging Face Transformers, and
PyTorch runtime instrumentation.

## 4. Architecture

ComputeOS consists of scheduling interfaces, runtime execution paths, telemetry,
benchmarking, replay, visualization, and research documentation.

## 5. Predictive Value Scheduling

PVS frames adaptive inference as an optimal stopping problem. At each runtime
decision point, the scheduler observes a `PVSFeatureVector` containing entropy,
uncertainty, activation norm, layer-to-layer variance, attention entropy,
decision pressure, normalized latency, normalized memory, and consumed compute
fraction. A lightweight value model estimates expected improvement, expected
compute cost, expected latency, expected memory, utility, cost, net value, and
stop probability. The runtime continues only when expected marginal value
exceeds the configured cost threshold and hard resource budgets remain feasible.

The oracle calibration script was run on three `distilgpt2` WikiText validation
prompts with three generated tokens each. Non-negative least-squares calibration
fit the following value weights: entropy 1.0000, uncertainty 0.0000, activation
norm 0.0000, layer variance 0.0000, attention entropy 0.0000, and decision
pressure 0.0000. With medium PVS budgets, the default weights measured 52.58 ms
mean latency and 15.918 mean perplexity, while the calibrated weights measured
50.32 ms mean latency and 8.113 mean perplexity on the same smoke run.

## 6. Counterfactual Runtime Intelligence

CRI replays completed telemetry traces and estimates alternative scheduling
trajectories without rerunning expensive inference. A `TraceLoader` converts
`ModelTelemetry` into deterministic replay events for request start, layer
completion, scheduler decisions, and request finish. `CounterfactualEngine`
then applies scenario transforms such as scheduler replacement or resource
budget changes and reports predicted latency, compute, memory, quality proxy,
regret, and oracle gap.

Validation compared CRI's static full-execution counterfactual against actual
controlled full-execution reruns for three `distilgpt2` WikiText prompts. The
measured latency MAE was 309.50 ms and the perplexity MAE was 7.683, failing the
predefined pass thresholds of 20 ms and 2.0 respectively. This failure is a
useful negative result: the current CRI proxy can replay observed telemetry, but
it is not yet calibrated enough to predict unobserved full-execution quality and
latency from early-stopped traces.

## 7. Oracle Scheduler

The oracle scheduler is an offline-only analysis tool that computes best
decisions under configurable objectives from completed traces. It enumerates
candidate stopping prefixes, scores each prefix under objectives such as
maximize quality, minimize latency, minimize FLOPs, minimize memory, balanced
utility, or maximize utility, and returns the best feasible plan under optional
latency, compute, and memory constraints. The oracle is intentionally not a live
`Scheduler`; it has access to completed traces and therefore acts as an analysis
upper bound rather than a deployable policy.

Oracle gap is defined as the oracle plan utility minus the online or
counterfactual utility for the same trace. In the calibration run, oracle
utilities supplied the target values for fitting PVS weights from recorded PVS
decision metadata. This makes the oracle a supervision source for future
learned schedulers while preserving the separation between offline analysis and
runtime scheduling.

## 8. Experiments

We ran a controlled Hugging Face sweep on `distilgpt2` using ten WikiText-103
validation prompts and twenty generated tokens per prompt. Conditions were run
in randomized order after one warm-up pass to eliminate JIT compilation bias.
The full baseline executed 120 transformer blocks per prompt on average, while
all PVS presets now triggered actual early exits and executed fewer layers than
baseline.

| Condition | Mean latency (ms) | Mean perplexity | Mean layers | Mean early exits | Latency reduction vs baseline | Perplexity delta |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 417.91 | 1.377 | 120.00 | 0.00 | 0.0% | 0.000 |
| pvs_loose | 71.39 | 4.971 | 20.00 | 1.00 | 82.9% | 3.594 |
| pvs_medium | 66.15 | 5.032 | 16.20 | 1.00 | 84.2% | 3.654 |
| pvs_tight | 173.46 | 5.498 | 12.00 | 1.00 | 58.5% | 4.121 |
| token_cap | 26.47 | 2.880 | 6.00 | 1.00 | 93.7% | 1.503 |

The calibrated compute budgets now exercise the controlled runtime rather than
only recording telemetry. `pvs_loose`, `pvs_medium`, and `pvs_tight` stop from
compute-budget exhaustion, reducing average executed layers from 120.00 to
20.00, 16.20, and 12.00 respectively. The `token_cap` preset triggers
value-based exits with the decision reason "expected net value below stopping
threshold" and reaches the most aggressive allocation, averaging 6.00 executed
layers and 93.7% lower measured latency than baseline.

Limitations of this benchmark: `distilgpt2` is tiny, so these results validate
the framework mechanics rather than production-scale serving behavior. Compute
units are abstract activation-size units, not measured FLOPs or hardware
counters. The perplexity deltas show a real quality/compute trade-off, and the
latency numbers should not be extrapolated to larger production models without
broader benchmarks, randomized repetitions, GPU measurements, and task-level
quality evaluation.

## 9. Ablations

Template only. Suggested ablations include removing PVS prediction features,
changing resource budgets, disabling telemetry, and comparing oracle gaps.

## 10. Limitations

The Hugging Face runtime is observational for arbitrary layer actions. True
FLOPs and Joules are estimated unless hardware counters are integrated.
Counterfactual outcomes are estimates unless validated by controlled runtime
experiments.

## 11. Future Work

Future work includes backend capability negotiation, learned counterfactual
models, broader benchmark suites, distributed runtime support, and interactive
visualization studio.

## References

See `references.bib`.
