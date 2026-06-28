# Research Roadmap

ComputeOS intentionally ships with only a minimal heuristic scheduler. The next
research milestones are:

1. **Confidence-aware early exit:** train a lightweight probe on hidden states and
   terminate generation steps when confidence exceeds a calibrated threshold.
2. **Layer skipping:** learn per-token skip decisions for middle transformer
   blocks, with guardrails for quality regression.
3. **Adaptive KV cache:** choose cache precision or eviction strategies based on
   prompt length, attention entropy, and available memory.
4. **RL scheduling:** formulate scheduling as a constrained MDP with latency,
   memory, and quality rewards.
5. **Distributed inference:** extend telemetry with rank-local metrics and
   policy coordination for tensor or pipeline parallel inference.

Each milestone should keep the current scheduler interface intact unless there
is evidence that a new execution boundary is required.

## Repository Hygiene

The public repository includes CI, an MIT license, and a tiny GPT-2 example.
Future pull requests should include focused unit tests for scheduler behavior
and should avoid requiring benchmark downloads in the default CI path.
