"""Inference execution pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.instrumentation.hooks import HookedTransformerMonitor
from computeos.scheduling.base import Scheduler
from computeos.telemetry.collector import TelemetryCollector
from computeos.telemetry.metrics import ModelTelemetry


@dataclass(frozen=True)
class ExecutionResult:
    """Generated text and telemetry for one inference request."""

    prompt: str
    generated_text: str
    telemetry: ModelTelemetry
    raw_outputs: dict[str, Any]


class InferenceEngine:
    """Coordinates tokenization, hook instrumentation, generation, and telemetry."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_name: str,
        scheduler: Scheduler,
        execution_config: ExecutionConfig,
        telemetry_config: TelemetryConfig,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._scheduler = scheduler
        self._execution_config = execution_config
        self._telemetry_config = telemetry_config

    @torch.inference_mode()
    def generate(self, prompt: str) -> ExecutionResult:
        """Run instrumented autoregressive generation for a prompt."""

        torch.manual_seed(self._execution_config.seed)
        self._scheduler.reset()
        collector = TelemetryCollector(model_name=self._model_name)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        started_at = perf_counter()
        with HookedTransformerMonitor(
            model=self._model,
            scheduler=self._scheduler,
            collector=collector,
            telemetry_config=self._telemetry_config,
        ):
            with _maybe_autocast(enabled=self._execution_config.autocast):
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._execution_config.max_new_tokens,
                    use_cache=self._execution_config.use_cache,
                    output_attentions=self._telemetry_config.capture_attention_entropy,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        total_latency_ms = (perf_counter() - started_at) * 1000.0
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        collector.record_confidence_scores(_token_confidence_scores(output_ids))
        peak_process_rss = max(
            (
                layer.process_rss_bytes
                for layer in collector.model_telemetry.layers
                if layer.process_rss_bytes
            ),
            default=None,
        )
        telemetry = collector.finish(
            total_latency_ms=total_latency_ms,
            peak_memory_bytes=peak_memory,
            peak_process_rss_bytes=peak_process_rss,
        )

        sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
        generated_text = self._tokenizer.decode(sequences[0], skip_special_tokens=True)
        return ExecutionResult(
            prompt=prompt,
            generated_text=generated_text,
            telemetry=telemetry,
            raw_outputs={"sequences": sequences.detach().cpu()},
        )


class _maybe_autocast:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled and torch.cuda.is_available()
        self._context: Any = None

    def __enter__(self) -> None:
        if self._enabled:
            self._context = torch.autocast(device_type="cuda")
            self._context.__enter__()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._context is not None:
            self._context.__exit__(exc_type, exc, traceback)


def _token_confidence_scores(output: Any) -> list[float]:
    scores = getattr(output, "scores", None)
    if not scores:
        return []
    confidences: list[float] = []
    for logits in scores:
        probs = torch.softmax(logits.detach().float(), dim=-1)
        confidences.extend(float(value.item()) for value in probs.max(dim=-1).values)
    return confidences
