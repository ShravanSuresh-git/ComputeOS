# ComputeOS Diagrams

## Overall Architecture

```mermaid
flowchart TD
    Config["Hydra / Dataclass Config"] --> Experiment["Experiment Runner"]
    Experiment --> Runtime["Runtime"]
    Experiment --> Benchmark["Benchmark"]
    Runtime --> Scheduler["Scheduler"]
    Runtime --> Telemetry["Telemetry"]
    Runtime --> Replay["Replay / CRI"]
    Telemetry --> Reports["Reports and Exports"]
```

## Runtime Lifecycle

```mermaid
sequenceDiagram
    participant Runtime
    participant Scheduler
    participant Layer
    participant Telemetry
    Runtime->>Scheduler: pre-layer decision
    Scheduler-->>Runtime: continue / skip / early exit
    Runtime->>Layer: execute if not skipped
    Layer-->>Runtime: output
    Runtime->>Telemetry: layer telemetry
    Runtime->>Scheduler: post-layer decision
    Scheduler-->>Runtime: continue / early exit
```

## Scheduler Lifecycle

```mermaid
flowchart LR
    Reset["reset"] --> Context["SchedulerContext"]
    Context --> Decide["decide"]
    Decide --> Decision["SchedulerDecision"]
    Decision --> Observe["observe"]
```

## Telemetry Pipeline

```mermaid
flowchart TD
    Layer["Layer Output"] --> Stats["Activation / Attention Stats"]
    Stats --> Collector["TelemetryCollector"]
    Collector --> Loggers["JSON / CSV / W&B"]
    Collector --> Reports["Terminal Reports"]
```

## Replay System

```mermaid
flowchart TD
    Telemetry["ModelTelemetry"] --> Loader["TraceLoader"]
    Loader --> Trace["ReplayTrace"]
    Trace --> Player["TracePlayer"]
    Trace --> Counterfactual["CounterfactualEngine"]
    Trace --> Oracle["OracleScheduler"]
    Counterfactual --> Regret["Regret Metrics"]
```

## Predictive Value Scheduling

```mermaid
flowchart LR
    Features["Runtime Features"] --> Prediction["Expected Improvement / Cost"]
    Prediction --> Net["Expected Net Value"]
    Net --> Continue["Continue"]
    Net --> Stop["Early Exit"]
```

## Counterfactual Runtime Intelligence

```mermaid
flowchart TD
    Trace["Completed Trace"] --> Scenario["Counterfactual Scenario"]
    Scenario --> Estimate["Predicted Outcome"]
    Trace --> Oracle["Oracle Plan"]
    Estimate --> Metrics["Utility / Regret / Gap"]
    Oracle --> Metrics
```

## Oracle Scheduler

```mermaid
flowchart LR
    Trace["Full Trace"] --> Objective["Objective"]
    Objective --> Search["Offline Stop Search"]
    Search --> Plan["Oracle Plan"]
```
