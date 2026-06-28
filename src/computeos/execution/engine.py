"""Inference execution pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution import BackendCapabilities
from computeos.instrumentation.hooks import HookedTransformerMonitor
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
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

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_early_exit=True,
            supports_token_level_control=True,
            supports_layer_level_control=False,
        )

    @torch.inference_mode()
    def generate(self, prompt: str) -> ExecutionResult:
        """Run instrumented autoregressive generation for a prompt."""

        torch.manual_seed(self._execution_config.seed)
        self._scheduler.reset()
        collector = TelemetryCollector(model_name=self._model_name)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = _model_device(self._model)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt_input_ids = inputs["input_ids"]
        input_ids = prompt_input_ids
        attention_mask = inputs.get("attention_mask", None)
        generated_ids: list[int] = []
        all_scores: list[torch.Tensor] = []
        log_prob_per_token: list[float] = []
        early_exit_applied = False
        started_at = perf_counter()
        with HookedTransformerMonitor(
            model=self._model,
            scheduler=self._scheduler,
            collector=collector,
            telemetry_config=self._telemetry_config,
            capabilities=self.capabilities,
        ):
            with _MaybeAutocast(enabled=self._execution_config.autocast):
                for step in range(self._execution_config.max_new_tokens):
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=self._execution_config.use_cache,
                        output_attentions=self._telemetry_config.capture_attention_entropy,
                        return_dict=True,
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    all_scores.append(next_token_logits.detach())
                    max_log_prob = torch.log_softmax(next_token_logits.float(), dim=-1).max()
                    log_prob_per_token.append(float(max_log_prob.item()))
                    next_token_id = int(next_token_logits.argmax(dim=-1).item())
                    generated_ids.append(next_token_id)

                    token_decision = self._scheduler.decide(
                        SchedulerContext(
                            step_index=step,
                            layer_name=None,
                            layer_telemetry=None,
                            model_telemetry=collector.model_telemetry,
                            backend_capabilities=self.capabilities,
                            metadata={"decision_point": "post_token", "token_index": step},
                        )
                    )

                    if token_decision.action == SchedulerAction.EARLY_EXIT:
                        token_decision = _with_action_result(
                            token_decision,
                            applied=True,
                            reason="token-level early exit",
                        )
                        early_exit_applied = True
                    collector.record_decision(token_decision)

                    if early_exit_applied:
                        break

                    eos_id = getattr(self._tokenizer, "eos_token_id", None)
                    if eos_id is not None and next_token_id == eos_id:
                        break

                    next_token_tensor = torch.tensor([[next_token_id]], device=input_ids.device)
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [
                                attention_mask,
                                torch.ones(
                                    (1, 1),
                                    device=attention_mask.device,
                                    dtype=attention_mask.dtype,
                                ),
                            ],
                            dim=-1,
                        )
        total_latency_ms = (perf_counter() - started_at) * 1000.0
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        collector.record_confidence_scores(_token_confidence_scores(all_scores))
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
        telemetry.metadata["early_exit_applied"] = early_exit_applied
        telemetry.metadata["tokens_generated"] = len(generated_ids)
        telemetry.metadata["log_prob_per_token"] = log_prob_per_token
        telemetry.metadata["all_scores_raw"] = log_prob_per_token

        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return ExecutionResult(
            prompt=prompt,
            generated_text=generated_text,
            telemetry=telemetry,
            raw_outputs={"sequences": _sequence_tensor(prompt_input_ids, generated_ids)},
        )


class _MaybeAutocast:
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


def _token_confidence_scores(scores: list[torch.Tensor]) -> list[float]:
    if not scores:
        return []
    confidences: list[float] = []
    for logits in scores:
        probs = torch.softmax(logits.detach().float(), dim=-1)
        confidences.extend(float(value.item()) for value in probs.max(dim=-1).values)
    return confidences


def _model_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _with_action_result(
    decision: SchedulerDecision,
    applied: bool,
    reason: str,
) -> SchedulerDecision:
    metadata = dict(decision.metadata)
    metadata["action_result"] = {"applied": applied, "reason": reason}
    return SchedulerDecision(
        action=decision.action,
        layer_name=decision.layer_name,
        confidence=decision.confidence,
        reason=decision.reason,
        metadata=metadata,
    )


def _sequence_tensor(prompt_input_ids: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
    if not generated_ids:
        return prompt_input_ids.detach().cpu()
    generated = torch.tensor([generated_ids], device=prompt_input_ids.device)
    return torch.cat([prompt_input_ids, generated], dim=-1).detach().cpu()
