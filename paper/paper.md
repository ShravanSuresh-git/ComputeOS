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

The oracle calibration was run on three WikiText-103 prompts with three
generated tokens per prompt, which is insufficient data to fit a non-degenerate
weight vector. The calibrated weights collapsed to `entropy: 1.0` and all other
feature weights at `0.0`, consistent with non-negative least squares overfitting
on a trivially small trace set. All experiments in Section 8 therefore use the
default hand-tuned weights: entropy 0.25, uncertainty 0.30, activation norm
0.15, layer variance 0.15, attention entropy 0.10, and decision pressure 0.05.
Calibration from longer traces using `calibrate_pvs_from_oracle.py` is left as
future work.

## 6. Counterfactual Runtime Intelligence

CRI replays completed telemetry traces and estimates alternative scheduling
trajectories without rerunning expensive inference. A `TraceLoader` converts
`ModelTelemetry` into deterministic replay events for request start, layer
completion, scheduler decisions, and request finish. `CounterfactualEngine`
then applies scenario transforms such as scheduler replacement or resource
budget changes and reports predicted latency, compute, memory, quality proxy,
regret, and oracle gap.

Validation compared CRI's static full-execution counterfactual against actual
controlled full-execution reruns for 20 `distilgpt2` reference-continuation
pairs. After the proportional latency estimator was added in v3, the latency
MAE decreased to 121.62 ms. The updated run still fails the predefined 20 ms
latency threshold, and the reference-perplexity MAE was 26.160, failing the 2.0
perplexity threshold. This failure is a useful negative result: the current CRI
proxy can replay observed telemetry and improves over the earlier 309.50 ms
latency MAE, but it is not yet calibrated enough to predict unobserved
full-execution quality and latency from early-stopped traces.

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

We ran a controlled Hugging Face sweep on `distilgpt2` using 50 reference
continuation pairs and twenty generated tokens per prompt. Conditions were run
in randomized order after one warm-up pass to eliminate JIT compilation bias.
The full baseline executed 120 transformer blocks per prompt on average, while
all PVS presets triggered actual early exits and executed fewer layers than
baseline.

| Condition | Mean latency (ms) | Reference perplexity | Mean layers | Mean early exits | Latency reduction vs baseline | PPL delta |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 365.77 +/- 87.45 | 54.38 +/- 48.85 | 120.00 | 0.00 | 0.0% | 0.000 |
| pvs_loose | 29.89 +/- 1.47 | 28.55 +/- 9.70 | 12.00 | 1.00 | 91.8% | -25.834 |
| pvs_medium | 29.76 +/- 1.48 | 28.55 +/- 9.70 | 12.00 | 1.00 | 91.9% | -25.834 |
| pvs_tight | 50.94 +/- 145.88 | 28.17 +/- 9.49 | 11.90 | 1.00 | 86.1% | -26.219 |
| token_cap | 18.13 +/- 1.27 | 26.76 +/- 8.50 | 6.00 | 1.00 | 95.0% | -27.626 |

| Condition | Oracle efficiency (%) | Mean oracle gap |
|---|---:|---:|
| baseline | 0.0% | 1.000 |
| pvs_loose | 99.9% | 0.001 |
| pvs_medium | 99.9% | 0.001 |
| pvs_tight | 100.0% | 0.000 |

The calibrated compute budgets now exercise the controlled runtime rather than
only recording telemetry. `pvs_loose`, `pvs_medium`, and `pvs_tight` reduce
average executed layers from 120.00 to 12.00, 12.00, and 11.90 respectively.
The `token_cap` preset triggers value-based exits with the decision reason
"expected net value below stopping threshold" and reaches the most aggressive
allocation, averaging 6.00 executed layers and 95.0% lower measured latency
than baseline. The Pareto frontier plot for this run was written to
`outputs/pareto_frontier.png`. Each point represents one scheduling condition;
perplexity is scored on reference continuation tokens given the text generated
by that condition's scheduler.

Perplexity is reference perplexity scored on held-out continuation tokens given
the text generated by each condition's scheduler, not self-scored generation
confidence. The negative PPL deltas in this controlled benchmark mean that the
shorter PVS-generated contexts happened to condition the curated continuation
task better than the full baseline generations. This should not be interpreted
as universal quality improvement; it shows why reference scoring must be
condition-specific and why broader task suites are needed.

The oracle-gap results are stronger for the system objective: `pvs_medium`
achieves 99.9% oracle efficiency versus 0.0% for the full baseline under the
current balanced utility proxy. This indicates that full inference wastes
substantial compute relative to the offline oracle on this trace distribution,
while PVS operates close to the oracle's preferred early-exit regime. The result
depends on the current utility model and abstract compute units, so it should be
treated as a framework validation target rather than a final claim about
production inference economics.

Limitations of this benchmark: `distilgpt2` is tiny, so these results validate
the framework mechanics rather than production-scale serving behavior. Compute
units are abstract activation-size units, not measured FLOPs or hardware
counters. The reference set is deterministic and designed for stable local
execution; results should not be extrapolated to production models without
broader benchmark suites, randomized repetitions, GPU measurements, and
task-level quality evaluation.

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
