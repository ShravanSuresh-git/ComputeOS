"""PyTorch hook manager for observing transformer execution."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import psutil
import torch
import torch.nn as nn

from computeos.config.schema import TelemetryConfig
from computeos.instrumentation.layers import discover_transformer_layers
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.telemetry.collector import TelemetryCollector
from computeos.telemetry.metrics import LayerTelemetry
from computeos.telemetry.stats import activation_stats, attention_entropy


class HookedTransformerMonitor:
    """Register hooks on transformer blocks and route observations to a scheduler."""

    def __init__(
        self,
        model: nn.Module,
        scheduler: Scheduler,
        collector: TelemetryCollector,
        telemetry_config: TelemetryConfig,
    ) -> None:
        self._model = model
        self._scheduler = scheduler
        self._collector = collector
        self._telemetry_config = telemetry_config
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._starts: dict[str, float] = {}
        self._step_index = 0
        self._process = psutil.Process()

    def __enter__(self) -> "HookedTransformerMonitor":
        self.register()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.remove()

    def register(self) -> None:
        layers = discover_transformer_layers(self._model)
        if not layers:
            layers = [(name, module) for name, module in self._model.named_children()]

        for name, module in layers:
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._make_post_hook(name)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._starts.clear()

    def _make_pre_hook(self, name: str) -> Any:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self._starts[name] = perf_counter()

        return hook

    def _make_post_hook(self, name: str) -> Any:
        def hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            started_at = self._starts.pop(name, perf_counter())
            latency_ms = (perf_counter() - started_at) * 1000.0
            layer_telemetry = LayerTelemetry(
                layer_name=name,
                layer_type=module.__class__.__name__,
                latency_ms=latency_ms,
                activation_stats=activation_stats(output)
                if self._telemetry_config.capture_activations
                else None,
                attention_entropy=attention_entropy(output)
                if self._telemetry_config.capture_attention_entropy
                else None,
                memory_allocated_bytes=_memory_allocated()
                if self._telemetry_config.capture_memory
                else None,
                memory_reserved_bytes=_memory_reserved()
                if self._telemetry_config.capture_memory
                else None,
                process_rss_bytes=int(self._process.memory_info().rss)
                if self._telemetry_config.capture_memory
                else None,
            )
            self._collector.record_layer(layer_telemetry)
            context = SchedulerContext(
                step_index=self._step_index,
                layer_name=name,
                layer_telemetry=layer_telemetry,
                model_telemetry=self._collector.model_telemetry,
            )
            decision = self._scheduler.decide(context)
            self._collector.record_decision(decision)
            self._scheduler.observe(context, decision)
            self._step_index += 1

        return hook


def _memory_allocated() -> int | None:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    return None


def _memory_reserved() -> int | None:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_reserved())
    return None
