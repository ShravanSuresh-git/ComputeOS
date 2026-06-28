---
name: Confidence-aware early exit scheduler
about: Track the first non-observational ComputeOS scheduling policy
title: "Implement confidence-aware early exit scheduler"
labels: enhancement, research
assignees: ""
---

## Goal

Add the first non-observational scheduling policy for ComputeOS: a
confidence-aware early exit scheduler.

## Proposed Scope

- Define confidence features available from hidden states, logits, or lightweight
  probes.
- Add a scheduler implementation behind the existing `Scheduler` API.
- Add Hydra config under `conf/scheduler/`.
- Add deterministic unit tests for decision thresholds and reset behavior.
- Add an experiment note documenting latency/quality tradeoffs on a small
  benchmark.

## Non-Goals

- Do not modify model weights in place.
- Do not couple the policy to one Hugging Face architecture unless the
  limitation is explicit.
- Do not require model downloads in the default CI path.
