# Runtime Enforcement

ComputeOS v1.0 has two runtime paths:

1. `InferenceEngine`
   - Uses Hugging Face `model.generate`.
   - Provides broad model compatibility.
   - Records scheduler decisions through hooks.
   - Does not apply arbitrary layer actions because `generate` owns the decode
     loop and model internals.

2. `ControlledForwardRuntime`
   - Executes an ordered layer sequence explicitly.
   - Applies scheduler decisions.
   - Supports `EARLY_EXIT`, `SKIP_LAYER`, and hard budgets.
   - Intended for research prototypes, tests, demos, and future model adapters.

## Supported Actions

| Action | Controlled runtime | Hugging Face generate runtime |
| --- | --- | --- |
| `CONTINUE` | applied | recorded |
| `RECORD_ONLY` | recorded | recorded |
| `EARLY_EXIT` | applied | recorded |
| `SKIP_LAYER` | applied at pre-layer boundary | recorded |
| `ADJUST_CACHE` | unsupported | recorded |

## Design Rationale

ComputeOS does not pretend that all runtimes can enforce all decisions. Runtime
capability must be explicit. The controlled runtime gives researchers a real
action-applying execution path without breaking the compatibility runtime.

## Example

```python
from computeos.execution.controlled import ControlledForwardRuntime

result = ControlledForwardRuntime(model, scheduler).run(inputs)
```

Applied action results are available in:

```python
result.action_results
result.telemetry.scheduler_decisions[*].metadata["action_result"]
```
