# Predictive Value Scheduling

Predictive Value Scheduling (PVS) is a mathematically grounded adaptive
inference algorithm for ComputeOS. It treats inference scheduling as an optimal
stopping problem: at every runtime decision point, the scheduler estimates the
marginal value of spending additional compute.

PVS asks:

> If I spend another unit of compute, how much improvement in answer quality
> should I expect?

It does not ask only whether another layer should run.

## 1. Mathematical Formulation

### State

At decision point `t`, the runtime state is:

```text
s_t = (x, y_<t, l_t, h_t, m_t, c_t, d_t)
```

where:

- `x` is the prompt
- `y_<t` is generated history
- `l_t` is the current layer or runtime boundary
- `h_t` summarizes activations
- `m_t` summarizes memory usage
- `c_t` summarizes compute already spent
- `d_t` summarizes historical scheduler decisions

In the current ComputeOS implementation, this state is approximated from
`SchedulerContext`, `LayerTelemetry`, and accumulated scheduler state.

### Action

The PVS action space is:

```text
a_t in {CONTINUE, EARLY_EXIT}
```

The current runtime records `EARLY_EXIT` decisions but does not yet apply them.
This is intentional: PVS is implemented as a scheduler plugin without modifying
runtime architecture. Future V2 backends can enforce the same decisions.

### Reward

PVS models reward as quality improvement minus resource cost:

```text
r_t = Delta Q_t - lambda_c C_t - lambda_l L_t - lambda_m M_t
```

where:

- `Delta Q_t` is expected quality improvement from additional compute
- `C_t` is expected compute cost
- `L_t` is expected latency cost
- `M_t` is expected memory cost
- `lambda_*` are configurable cost weights

### Value Function

The marginal value function is:

```text
V(s_t) = E[Delta Q_t | s_t] - E[Cost_t | s_t]
```

Continue execution only when:

```text
V(s_t) > tau
```

where `tau` is `budgets.min_net_value`.

### Compute Budget

PVS uses explicit resource budgets:

```text
B = (B_latency, B_memory, B_compute)
```

The scheduler stops when a budget is exhausted or when expected net value is
not positive enough.

### Stopping Criterion

PVS emits `EARLY_EXIT` when any condition holds:

```text
cumulative_latency >= B_latency
cumulative_memory >= B_memory
cumulative_compute >= B_compute
V(s_t) <= tau
```

### Constraints

- Schedulers do not execute inference.
- Decisions are immutable.
- Every prediction and decision is logged.
- Unsupported actions remain auditable metadata until runtime backends can
  apply them.

## 2. Runtime Architecture

PVS plugs into the existing ComputeOS scheduler interface:

```text
HookedTransformerMonitor
  -> LayerTelemetry
  -> SchedulerContext
  -> PredictiveValueScheduler.decide
  -> SchedulerDecision(metadata=prediction_trace)
  -> TelemetryCollector
```

No runtime architecture is modified.

### Telemetry Consumption

PVS consumes:

- activation norm
- activation variance across layers
- attention entropy when available
- token confidence when available
- layer latency
- memory usage
- historical scheduler decision count
- cumulative compute units

### Prediction

The scheduler constructs a normalized feature vector:

```text
phi(s_t) = [
  entropy,
  uncertainty,
  activation_norm,
  layer_variance,
  attention_entropy,
  decision_pressure,
  normalized_latency,
  normalized_memory,
  compute_fraction
]
```

It then estimates:

- expected improvement
- expected compute cost
- expected latency
- expected memory
- expected utility
- expected cost
- expected net value

### Execution Decisions

PVS returns:

- `CONTINUE` when expected marginal value exceeds cost
- `EARLY_EXIT` when expected net value is below threshold or a budget is
  exhausted
- `RECORD_ONLY` after a stopping event because the current ComputeOS runtime is
  observational

## 3. Prediction Model

The first PVS implementation uses a deterministic lightweight value model:

```text
E[Delta Q | s] =
  w_e entropy
  + w_u uncertainty
  + w_n activation_norm
  + w_v layer_variance
  + w_a attention_entropy
  + w_d decision_pressure
```

Expected cost is:

```text
E[Cost | s] =
  alpha_c compute_fraction
  + alpha_l normalized_latency
  + alpha_m normalized_memory
```

This model is deliberately simple. It is not a random heuristic:

- it encodes a marginal value objective
- all coefficients are explicit and configurable
- every prediction is logged
- future learned regressors can replace the estimator behind the same API

## 4. Scheduling Policy

For each decision:

```text
expected_utility = utility_scale * expected_improvement
expected_cost = cost_scale * resource_cost
expected_net_value = expected_utility - expected_cost
```

Policy:

```text
if budget exhausted:
    EARLY_EXIT
elif expected_net_value <= min_net_value:
    EARLY_EXIT
else:
    CONTINUE
```

## 5. Plugin Implementation

PVS is implemented as:

- `computeos.scheduling.pvs.PredictiveValueScheduler`
- Hydra config: `conf/scheduler/pvs.yaml`
- Registry name: `pvs`

Example:

```bash
computeos-run scheduler=pvs model.name=sshleifer/tiny-gpt2 execution.max_new_tokens=8
```

Python example:

```bash
python examples/run_pvs_tiny_gpt2.py
```

## 6. Telemetry and Replay

Every PVS decision writes metadata:

- algorithm name
- feature vector
- prediction
- budgets
- cumulative latency
- cumulative compute
- peak memory
- stopping event flag

Replay support:

- call `PredictiveValueScheduler.replay()` during an in-process run
- use `computeos.visualization.pvs.extract_pvs_trace(telemetry)` from logged
  telemetry

## 7. Benchmarks

PVS should be compared against:

- static execution
- entropy scheduling
- confidence scheduling
- random scheduling

Metrics:

- accuracy or task score
- latency
- estimated FLOPs or compute units
- memory
- utility
- budget efficiency
- scheduler regret

Regret can be estimated as:

```text
Regret = Utility_oracle - Utility_policy
```

where the oracle is approximated by retrospective full-execution telemetry.

## 8. Visualization

PVS includes optional visualization helpers:

```python
from computeos.visualization.pvs import plot_pvs_trace

plot_pvs_trace(result["telemetry"], "outputs/pvs_timeline.png")
```

The plot shows:

- expected improvement
- expected cost
- expected net value
- stopping decisions

Future visualizations should add:

- runtime timeline
- compute allocation
- memory pressure
- budget consumption
- utility heatmaps by token/layer

## 9. Ablation Studies

Recommended ablations:

1. without prediction model: constant improvement estimate
2. without stopping criterion: always continue
3. without telemetry: feature vector replaced with zeros
4. different value functions: linear, logistic, calibrated regressor
5. different resource budgets: latency-only, memory-only, compute-only, mixed
6. different cost scales
7. with and without attention entropy
8. with and without activation variance

## 10. Assumptions and Limitations

Current limitations:

- `EARLY_EXIT` is recorded but not enforced by the existing runtime.
- Confidence is usually post-hoc in the current Hugging Face `generate` path.
- Estimated compute units are activation-size proxies, not true FLOPs.
- The value model is hand-specified and should be calibrated on traces.
- Attention entropy is available only when model outputs expose attention
  probabilities at the hook boundary.

## Future Work

- train calibrated improvement predictors from full-execution traces
- add oracle regret estimation
- add controlled runtime backend that applies `EARLY_EXIT`
- add entropy/confidence/random baseline schedulers
- add publication-quality plots and LaTeX tables
- integrate PVS into benchmark comparison reports
