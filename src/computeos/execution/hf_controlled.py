"""Controlled Hugging Face decoder-only execution backend.

This backend executes transformer blocks explicitly so scheduler actions can be
applied to real Hugging Face causal language models. It is intentionally narrow:
GPT-2/distilgpt2 style models are fully supported, and LLaMA-style models are
validated as decoder-only but currently rejected with a clear error until a
position-embedding/RoPE adapter is added.
"""

from __future__ import annotations

import uuid as _uuid
from collections.abc import Callable
from time import perf_counter
from typing import Any, Protocol, cast

import psutil
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution import BackendCapabilities
from computeos.execution.engine import ExecutionResult
from computeos.instrumentation.layers import discover_transformer_layers
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.telemetry.collector import TelemetryCollector
from computeos.telemetry.metrics import LayerTelemetry
from computeos.telemetry.stats import activation_stats, attention_entropy


class _TokenizerLike(Protocol):
    eos_token_id: int | None

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        """Tokenize a single prompt."""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str | list[str]:
        """Decode generated token ids."""


class HFControlledEngine:
    """Greedy controlled decoder for Hugging Face causal language models.

    The engine mirrors :class:`computeos.execution.engine.InferenceEngine` at the
    public boundary but bypasses ``model.generate()``. Each token is produced by
    running embedding, transformer block, final norm, and LM-head stages
    directly. Before every transformer block, the scheduler is consulted and
    ``SKIP_LAYER``/``EARLY_EXIT`` decisions are enforced when possible.
    """

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
        self._tokenizer = cast(_TokenizerLike, tokenizer)
        self._model_name = model_name
        self._scheduler = scheduler
        self._execution_config = execution_config
        self._telemetry_config = telemetry_config
        self._adapter = _build_decoder_adapter(model)
        self._process = psutil.Process()

    @property
    def capabilities(self) -> BackendCapabilities:
        """Report runtime control features available to schedulers."""

        return BackendCapabilities(
            supports_early_exit=True,
            supports_skip_layer=True,
            supports_adjust_cache=False,
            supports_token_level_control=True,
            supports_layer_level_control=True,
        )

    @torch.inference_mode()
    def generate(self, prompt: str) -> ExecutionResult:
        """Generate text with greedy decoding and controlled layer execution."""

        torch.manual_seed(self._execution_config.seed)
        self._scheduler.reset()
        cast(nn.Module, self._model).eval()
        collector = TelemetryCollector(
            model_name=self._model_name,
            request_id=str(_uuid.uuid4()),
        )
        device = _model_device(self._model)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        prompt_input_ids = input_ids
        generated_ids: list[int] = []
        all_scores: list[torch.Tensor] = []
        log_prob_per_token: list[float] = []
        previous_layer_telemetry: dict[str, LayerTelemetry] = {}
        early_exit_applied = False
        early_exit_layer: str | None = None
        compute_units = 0.0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        started_at = perf_counter()
        with _MaybeAutocast(enabled=self._execution_config.autocast):
            for token_index in range(self._execution_config.max_new_tokens):
                hidden_states = self._adapter.embed(input_ids)
                for layer_index, (layer_name, block) in enumerate(self._adapter.layers):
                    decision = self._scheduler.decide(
                        SchedulerContext(
                            step_index=len(collector.model_telemetry.scheduler_decisions),
                            layer_name=layer_name,
                            layer_telemetry=previous_layer_telemetry.get(layer_name),
                            model_telemetry=collector.model_telemetry,
                            metadata={
                                "decision_point": "pre_layer",
                                "token_index": token_index,
                                "layer_index": layer_index,
                            },
                            backend_capabilities=self.capabilities,
                        )
                    )
                    if decision.action == SchedulerAction.SKIP_LAYER:
                        collector.record_decision(
                            _with_action_result(decision, True, "layer skipped")
                        )
                        self._scheduler.observe(
                            _observe_context(
                                layer_name=layer_name,
                                layer_telemetry=previous_layer_telemetry.get(layer_name),
                                collector=collector,
                                capabilities=self.capabilities,
                            ),
                            decision,
                        )
                        continue
                    if decision.action == SchedulerAction.EARLY_EXIT:
                        collector.record_decision(
                            _with_action_result(decision, True, "early exit before layer")
                        )
                        early_exit_applied = True
                        early_exit_layer = layer_name
                        break
                    collector.record_decision(_with_action_result(decision, False, "recorded"))

                    layer_started_at = perf_counter()
                    block_output = block(
                        hidden_states,
                        use_cache=False,
                        output_attentions=self._telemetry_config.capture_attention_entropy,
                    )
                    hidden_states = _extract_hidden_states(block_output)
                    latency_ms = (perf_counter() - layer_started_at) * 1000.0
                    raw_entropy = (
                        attention_entropy(block_output)
                        if self._telemetry_config.capture_attention_entropy
                        else None
                    )
                    layer_telemetry = LayerTelemetry(
                        layer_name=layer_name,
                        layer_type=block.__class__.__name__,
                        latency_ms=latency_ms,
                        activation_stats=activation_stats(hidden_states)
                        if self._telemetry_config.capture_activations
                        else None,
                        attention_entropy=raw_entropy,
                        attention_entropy_available=raw_entropy is not None,
                        memory_allocated_bytes=_memory_allocated()
                        if self._telemetry_config.capture_memory
                        else None,
                        memory_reserved_bytes=_memory_reserved()
                        if self._telemetry_config.capture_memory
                        else None,
                        process_rss_bytes=int(self._process.memory_info().rss)
                        if self._telemetry_config.capture_memory
                        else None,
                        metadata={"token_index": token_index, "layer_index": layer_index},
                    )
                    previous_layer_telemetry[layer_name] = layer_telemetry
                    compute_units += _compute_units(layer_telemetry)
                    collector.record_layer(layer_telemetry)
                    self._scheduler.observe(
                        _observe_context(
                            layer_name=layer_name,
                            layer_telemetry=layer_telemetry,
                            collector=collector,
                            capabilities=self.capabilities,
                        ),
                        decision,
                    )

                hidden_states = self._adapter.final_norm(hidden_states)
                next_token_logits = self._adapter.lm_head(hidden_states)[:, -1, :]
                all_scores.append(next_token_logits.detach())
                live_confidence = float(
                    torch.softmax(next_token_logits.detach().float(), dim=-1).max().item()
                )
                collector.push_confidence_score(live_confidence)
                max_log_prob = torch.log_softmax(next_token_logits.float(), dim=-1).max()
                log_prob_per_token.append(float(max_log_prob.item()))
                next_token_id = int(next_token_logits.argmax(dim=-1).item())
                generated_ids.append(next_token_id)

                if early_exit_applied:
                    break
                eos_id = self._tokenizer.eos_token_id
                if eos_id is not None and next_token_id == eos_id:
                    break
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token_id]], device=device)],
                    dim=-1,
                )

        total_latency_ms = (perf_counter() - started_at) * 1000.0
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        peak_process_rss = max(
            (
                layer.process_rss_bytes
                for layer in collector.model_telemetry.layers
                if layer.process_rss_bytes is not None
            ),
            default=None,
        )
        telemetry = collector.finish(
            total_latency_ms=total_latency_ms,
            peak_memory_bytes=peak_memory,
            peak_process_rss_bytes=peak_process_rss,
        )
        telemetry.metadata["runtime"] = "hf_controlled"
        telemetry.metadata["early_exit_applied"] = early_exit_applied
        telemetry.metadata["early_exit_layer"] = early_exit_layer
        telemetry.metadata["tokens_generated"] = len(generated_ids)
        telemetry.metadata["token_indices"] = list(range(len(generated_ids)))
        telemetry.metadata["log_prob_per_token"] = log_prob_per_token
        telemetry.metadata["all_scores_raw"] = log_prob_per_token
        telemetry.metadata["compute_units"] = compute_units

        generated_text = _decode_text(self._tokenizer, generated_ids)
        return ExecutionResult(
            prompt=prompt,
            generated_text=generated_text,
            telemetry=telemetry,
            raw_outputs={
                "sequences": _sequence_tensor(prompt_input_ids, generated_ids),
                "scores": [score.detach().cpu() for score in all_scores],
            },
        )

    def warm_up(self, prompt: str | None = None) -> None:
        """Run warm-up passes before timed measurements."""

        warmup_prompt = prompt or "warm up"
        for _ in range(self._execution_config.warmup_runs):
            self.generate(warmup_prompt)


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


class _DecoderAdapter:
    def __init__(
        self,
        layers: list[tuple[str, nn.Module]],
        embed: Callable[[torch.Tensor], torch.Tensor],
        final_norm: Callable[[torch.Tensor], torch.Tensor],
        lm_head: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.layers = layers
        self.embed = embed
        self.final_norm = final_norm
        self.lm_head = lm_head


def _build_decoder_adapter(model: PreTrainedModel) -> _DecoderAdapter:
    config = getattr(model, "config", None)
    if bool(getattr(config, "is_encoder_decoder", False)):
        raise ValueError("HFControlledEngine only supports decoder-only causal language models.")

    layers = discover_transformer_layers(model)
    if not layers:
        raise ValueError(
            "HFControlledEngine could not find transformer layers. "
            "Expected GPT-2/distilgpt2 blocks such as transformer.h.0."
        )

    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "wte") and hasattr(transformer, "h"):
        layer_modules = [
            (f"transformer.h.{index}", block)
            for index, block in enumerate(transformer.h)
        ]

        def embed(input_ids: torch.Tensor) -> torch.Tensor:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            token_embeddings = cast(torch.Tensor, transformer.wte(input_ids))
            position_embeddings = cast(torch.Tensor, transformer.wpe(position_ids))
            hidden_states = token_embeddings + position_embeddings
            drop = getattr(transformer, "drop", None)
            if isinstance(drop, nn.Module):
                hidden_states = cast(torch.Tensor, drop(hidden_states))
            return hidden_states

        def final_norm(hidden_states: torch.Tensor) -> torch.Tensor:
            ln_f = getattr(transformer, "ln_f", None)
            if isinstance(ln_f, nn.Module):
                return cast(torch.Tensor, ln_f(hidden_states))
            return hidden_states

        lm_head = getattr(model, "lm_head", None)
        if not isinstance(lm_head, nn.Module):
            raise ValueError("HFControlledEngine requires a decoder-only model with an lm_head.")
        return _DecoderAdapter(layer_modules, embed, final_norm, lm_head)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        raise ValueError(
            "LLaMA-style decoder layers were discovered, but this adapter has not "
            "implemented RoPE/cache handling yet. Use GPT-2 or distilgpt2 for controlled runs."
        )

    raise ValueError(
        "HFControlledEngine currently supports GPT-2/distilgpt2-style decoder-only models."
    )


def _extract_hidden_states(block_output: object) -> torch.Tensor:
    if isinstance(block_output, torch.Tensor):
        return block_output
    if isinstance(block_output, (tuple, list)) and block_output:
        first = block_output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError("Transformer block output did not contain hidden states as its first tensor.")


def _observe_context(
    layer_name: str,
    layer_telemetry: LayerTelemetry | None,
    collector: TelemetryCollector,
    capabilities: BackendCapabilities,
) -> SchedulerContext:
    return SchedulerContext(
        step_index=len(collector.model_telemetry.scheduler_decisions),
        layer_name=layer_name,
        layer_telemetry=layer_telemetry,
        model_telemetry=collector.model_telemetry,
        backend_capabilities=capabilities,
    )


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


def _compute_units(layer: LayerTelemetry) -> float:
    if layer.activation_stats is None:
        return 1.0
    return max(1.0, layer.activation_stats.numel / 1024.0)


def _memory_allocated() -> int | None:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    return None


def _memory_reserved() -> int | None:
    if torch.cuda.is_available():
        return int(torch.cuda.memory_reserved())
    return None


def _sequence_tensor(prompt_input_ids: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
    if not generated_ids:
        return prompt_input_ids.detach().cpu()
    generated = torch.tensor([generated_ids], device=prompt_input_ids.device)
    return torch.cat([prompt_input_ids, generated], dim=-1).detach().cpu()


def _decode_text(tokenizer: _TokenizerLike, token_ids: list[int]) -> str:
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    if isinstance(decoded, list):
        return "".join(decoded)
    return decoded
