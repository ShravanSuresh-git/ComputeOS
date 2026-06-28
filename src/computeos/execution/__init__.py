"""Model loading and inference execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


@dataclass(frozen=True)
class BackendCapabilities:
    """Runtime control features supported by an execution backend."""

    supports_early_exit: bool = False
    supports_skip_layer: bool = False
    supports_adjust_cache: bool = False
    supports_token_level_control: bool = False
    supports_layer_level_control: bool = False


if TYPE_CHECKING:
    from computeos.execution.controlled import (
        ActionResult,
        ControlledExecutionResult,
        ControlledForwardRuntime,
        RuntimeBudget,
    )
    from computeos.execution.engine import ExecutionResult, InferenceEngine
    from computeos.execution.model_loader import LoadedModel

__all__ = [
    "ActionResult",
    "BackendCapabilities",
    "ControlledExecutionResult",
    "ControlledForwardRuntime",
    "ExecutionResult",
    "InferenceEngine",
    "LoadedModel",
    "RuntimeBudget",
    "load_hf_causal_lm",
]


def __getattr__(name: str) -> object:
    controlled_exports = {
        "ActionResult",
        "ControlledExecutionResult",
        "ControlledForwardRuntime",
        "RuntimeBudget",
    }
    if name in controlled_exports:
        from computeos.execution import controlled

        return getattr(controlled, name)
    if name in {"ExecutionResult", "InferenceEngine"}:
        from computeos.execution import engine

        return getattr(engine, name)
    if name in {"LoadedModel", "load_hf_causal_lm"}:
        from computeos.execution import model_loader

        return getattr(model_loader, name)
    raise AttributeError(f"module 'computeos.execution' has no attribute {name!r}")
