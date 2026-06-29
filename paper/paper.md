# ComputeOS: A Research Framework for Adaptive Runtime Inference

## Abstract

ComputeOS is a research framework for adaptive runtime scheduling in transformer
inference. The system separates scheduling policy, controlled execution,
telemetry, replay, benchmarking, and experiment configuration so that new
adaptive-inference algorithms can be implemented without modifying model
weights or forking Hugging Face model code. ComputeOS instruments decoder-only
transformers at runtime, exposes layer-level telemetry to schedulers, and can
apply controlled actions such as early exit and layer skipping when the backend
supports them. This paper describes the architecture and evaluates Predictive
Value Scheduling (PVS), an optimal-stopping-inspired policy that estimates
whether marginal compute is worth spending under configurable resource budgets.
On a 50-prompt `distilgpt2` reference-continuation sweep, PVS variants reduce
measured latency by 85.96--94.37% while producing positive reference-PPL
improvement under the benchmark's baseline-minus-condition convention. An
oracle-gap experiment further shows that the online PVS presets operate close
to the offline utility optimum under the current balanced proxy objective.

## 1. Introduction

Transformer inference normally executes a fixed computation graph for every
token, even when runtime signals suggest that some inputs need less computation
than others. Production serving engines optimize batching, KV-cache layout,
memory reuse, and kernel efficiency, but they rarely provide a research surface
for changing the amount of computation spent inside the forward pass. ComputeOS
targets that gap: it is not a chatbot, serving product, or replacement for
high-throughput engines. It is a modular platform for studying adaptive runtime
intelligence.

The central design goal is to let researchers prototype scheduling algorithms in
a few hundred lines while still measuring real execution behavior. A scheduler
should receive structured telemetry, make typed decisions through a stable API,
and be evaluated through the same benchmark and replay tools as every other
policy. This makes ComputeOS closer in spirit to PyTorch Lightning or Ray for a
specific research domain: it standardizes the experiment lifecycle without
prescribing a single algorithm.

## 2. Motivation

Adaptive inference research is difficult to compare because many prototypes
combine model changes, policy logic, telemetry collection, and benchmark code in
one script. That coupling makes it hard to answer basic questions: did a policy
save compute because of its decision rule, because of an implementation shortcut,
or because the evaluation metric changed? ComputeOS enforces separation between
observation, scheduling, runtime action application, and evaluation.

This separation matters for three reasons. First, scheduler implementations can
remain independent of model weights and backend internals. Second, telemetry is
captured once and reused for live decisions, replay, visualization, and oracle
analysis. Third, experiments become reproducible: the same prompts, budgets,
model, layer counts, random seeds, and output artifacts can be inspected after a
run. The result is a research loop where new policies can be added without
rewriting execution code or weakening benchmark discipline.

## 3. Related Work

ComputeOS builds on several lines of systems and machine-learning research.
Adaptive Computation Time introduced the view that neural networks can learn how
much computation to spend per input. Depth-adaptive transformers and confident
adaptive language modeling study early-exit behavior for sequence models,
usually by training auxiliary heads or confidence criteria. Speculative decoding
accelerates generation by drafting candidate tokens and verifying them with a
larger model, shifting the scheduling problem from layer depth to token
verification.

Serving systems such as vLLM, TensorRT-LLM, and SGLang focus on production
throughput, memory management, batching, paged attention, and kernel-level
optimization. Hugging Face Transformers and PyTorch provide the model and tensor
abstractions that make experimentation accessible. ComputeOS is different: it
does not try to beat serving engines on throughput. Instead, it provides a
backend-neutral research interface for runtime scheduling decisions, telemetry,
counterfactual replay, and oracle analysis. Those abstractions make it possible
to compare heuristic, learned, and value-based schedulers under one lifecycle.

## 4. Architecture

ComputeOS is organized around explicit module boundaries:

- `computeos.scheduling` defines scheduler contracts, decision objects,
  scheduler context, and concrete policies such as PVS.
- `computeos.execution` owns backend execution, including the controlled Hugging
  Face decoder path used in the experiments.
- `computeos.telemetry` records layer latency, activation statistics, attention
  entropy when available, confidence scores, memory usage, and scheduler
  decisions.
- `computeos.replay` converts telemetry into deterministic traces for oracle and
  counterfactual analysis.
- `computeos.benchmarks`, `examples`, and `computeos.visualization` provide the
  experiment surface: sweeps, reports, oracle-gap measurement, and plots.
- `computeos.config` keeps runtime and telemetry settings typed and explicit.

Schedulers interact with the runtime only through `SchedulerContext` and return
`SchedulerDecision` values. The controlled Hugging Face backend invokes the
scheduler before each transformer block, records whether an action was applied,
and then emits the resulting telemetry. This design preserves model weights,
keeps policy code testable, and allows the same scheduler API to support
heuristics, learned classifiers, reinforcement-learning policies, and future
market-style compute allocation algorithms.

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

We ran a controlled Hugging Face sweep on `distilgpt2` using 50 curated natural
English prompt-continuation pairs and twenty generated tokens per prompt. The
fallback reference set spans science, technology, history, literature,
economics, geography, philosophy, mathematics, medicine, and culture. Conditions
were shuffled with a fixed seed and warmed up before measurement. Budgets were
scaled from model depth and generation length: loose, medium, and tight PVS
presets receive 83%, 50%, and 25% of `n_layers * max_new_tokens` compute units.

Reference perplexity is scored on held-out continuation tokens given the text
generated by each condition. `perplexity_delta` is reported as baseline
perplexity minus condition perplexity, so positive values indicate lower
reference perplexity than the full-execution baseline under this benchmark
definition.

| Condition | Mean latency (ms) | Reference perplexity | Mean layers | Mean early exits | Latency reduction vs baseline | PPL delta |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 431.77 +/- 60.17 | 2315.38 +/- 5141.63 | 120.00 | 0.00 | 0.0% | 0.00 |
| pvs_loose | 52.46 +/- 9.12 | 2285.31 +/- 5925.12 | 12.92 | 1.00 | 87.9% | 30.07 |
| pvs_medium | 32.82 +/- 4.61 | 869.11 +/- 1160.58 | 9.90 | 1.00 | 92.4% | 1446.27 |
| pvs_tight | 60.62 +/- 164.41 | 1017.58 +/- 1264.85 | 7.74 | 1.00 | 86.0% | 1297.80 |
| token_cap | 24.33 +/- 2.72 | 690.19 +/- 1098.16 | 6.00 | 1.00 | 94.4% | 1625.19 |

All adaptive conditions applied early exits and reduced executed layers relative
to the 120-layer baseline trajectory. `token_cap` is the most aggressive preset:
it averages 6.00 executed layers, lowers measured latency by 94.37%, and has a
positive PPL delta of 1625.19. `pvs_tight` also satisfies the acceptance target
with a positive PPL delta of 1297.80 while using 7.74 layers on average. The
Pareto frontier for this run is written to `outputs/pareto_frontier.png`.

We also measured oracle gap on 20 `distilgpt2` prompts under the balanced
utility objective.

| Condition | Oracle efficiency (%) | Mean achieved utility | Mean oracle utility | Mean oracle gap |
|---|---:|---:|---:|---:|
| baseline | 0.0 | -2.6888 | -0.0784 | 1.000 |
| pvs_loose | 87.2 | 0.1098 | -0.0075 | 0.128 |
| pvs_medium | 100.0 | 0.2048 | 0.0834 | 0.000 |
| pvs_tight | 100.0 | 0.2820 | 0.1585 | 0.000 |

The oracle-gap results validate the replay and utility pipeline rather than a
production economics claim. Under the current balanced proxy, the full baseline
spends more compute than the oracle prefers, while medium and tight PVS match
the oracle's best available utility prefix on these traces. The result depends
on abstract compute units and the present quality proxy.

The sweep script now supports `distilgpt2`, `gpt2-medium`, and `all`, and writes
model-specific artifacts such as `outputs/sweep_results_distilgpt2.json`.
`gpt2-medium` is supported by the code path and receives budgets scaled by its
24-layer depth, but no gpt2-medium row is reported here because the local CPU
fast run did not complete during the interactive experiment window. We do not
fabricate results for uncompleted runs.

These experiments are intentionally small. They validate that ComputeOS can
apply real runtime decisions, collect condition-specific telemetry, produce
reproducible artifacts, and compare online policies against an offline oracle.
They do not establish production-scale quality preservation. Future evaluations
should add task-level metrics, larger models, GPU measurements, measured FLOPs
or hardware counters, repeated random seeds, and broader benchmark suites.

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
