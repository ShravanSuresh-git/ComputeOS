# Counterfactual Runtime Intelligence

Counterfactual Runtime Intelligence (CRI) is ComputeOS' offline replay and
counterfactual evaluation subsystem.

Traditional adaptive inference evaluation asks:

> What happened under the policy we executed?

CRI asks:

> What would have happened under a different scheduler, budget, or stopping
> criterion?

CRI does not rerun expensive inference. It reuses completed ComputeOS telemetry
and estimates alternative trajectories with explicit proxy models that can be
replaced by future learned predictors.

## Architecture

```text
ModelTelemetry
  -> TraceLoader
  -> ReplayTrace
  -> TracePlayer
  -> CounterfactualEngine
  -> OracleScheduler
  -> Regret + Metrics
  -> Experiment Tables + Visualizations
```

The package lives under `computeos.replay`:

- `trace_loader.py`: converts completed telemetry into replay traces
- `trace_player.py`: deterministic pause/resume/step/seek replay
- `counterfactual_engine.py`: evaluates alternative trajectories
- `oracle_scheduler.py`: offline-only oracle policy
- `regret.py`: scheduler regret metrics
- `metrics.py`: research metrics
- `visualizer.py`: timeline and regret plots
- `scenario.py`: counterfactual scenario definitions
- `experiment.py`: policy comparison tables and exports

## Mathematical Formulation

Given a completed trace:

```text
T = (e_1, e_2, ..., e_n)
```

CRI estimates the outcome of a counterfactual policy `pi'` without rerunning the
model:

```text
Y_hat(pi' | T) = f(T, pi', B, theta)
```

where:

- `T` is observed telemetry
- `pi'` is an alternative policy
- `B` is a resource budget
- `theta` are estimator parameters

Utility is represented as:

```text
U = Q_hat - alpha L - beta C - gamma M
```

where:

- `Q_hat` is a quality proxy or benchmark score
- `L` is latency
- `C` is estimated compute
- `M` is memory

CRI reports scheduler regret:

```text
Regret = U_oracle - U_policy
```

## Oracle Scheduler

The `OracleScheduler` has access to the complete trace and is strictly
offline-only. It does not implement the live scheduler interface.

Objectives:

- maximize quality
- minimize latency
- minimize FLOPs
- minimize memory
- maximize utility
- balanced multi-objective optimization

The oracle is used to estimate upper bounds and regret, not to run live
inference.

## Counterfactual Scenarios

Supported scenarios include:

- continue extra layers
- increase reasoning budget
- decrease reasoning budget
- change latency budget
- change compute budget
- change memory budget
- replace scheduler
- change stopping criterion

Example:

```python
from computeos.replay import CounterfactualEngine, CounterfactualScenario, ScenarioType

scenario = CounterfactualScenario(
    name="continue_5_more_layers",
    scenario_type=ScenarioType.CONTINUE_EXTRA_LAYERS,
    extra_layers=5,
)
result = CounterfactualEngine().evaluate(trace, scenario)
```

## Replay

`TracePlayer` supports:

- pause
- resume
- step
- seek
- adjustable speed
- deterministic event iteration

The replay engine exposes:

- scheduler decisions
- runtime state
- telemetry
- latency
- memory
- utility
- confidence
- activation statistics

## Metrics

CRI implements:

- scheduler regret
- budget efficiency
- expected utility
- expected information gain
- utility per FLOP
- utility per Joule
- counterfactual improvement
- decision stability
- stopping accuracy
- oracle gap

## Benchmark Integration

`CounterfactualExperiment.default_policy_comparison` compares offline proxies
for:

- static scheduler
- entropy scheduler
- confidence scheduler
- random scheduler
- PVS
- oracle scheduler

It exports:

- JSON
- CSV
- Markdown
- LaTeX

These are counterfactual estimates, not measured experimental results. They are
useful for debugging and research planning until controlled runtimes can apply
all actions live.

## Visualization

`ReplayVisualizer` can generate:

- runtime timeline
- layer execution plot
- scheduler decision plot
- regret timeline

Future versions should add interactive replay, budget heatmaps, oracle decision
overlays, and branch visualizations.

## Limitations

- CRI estimates counterfactual outcomes; it does not prove them.
- Quality is a proxy unless a benchmark score is attached.
- True FLOPs and Joules are not available in current telemetry.
- Oracle decisions are offline upper bounds and must never be used live.
- Counterfactual scheduler replacements are proxy models until each scheduler
  has an offline replay adapter.

## Future Work

- learned counterfactual predictors
- calibrated oracle regret
- interactive web replay studio
- branch comparison visualizations
- integration with benchmark score artifacts
- controlled runtime validation for CRI estimates
