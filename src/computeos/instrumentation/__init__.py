"""PyTorch hook-based instrumentation."""

from computeos.instrumentation.hooks import HookedTransformerMonitor
from computeos.instrumentation.layers import discover_transformer_layers

__all__ = ["HookedTransformerMonitor", "discover_transformer_layers"]
