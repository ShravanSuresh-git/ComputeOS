"""Hugging Face model loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from computeos.config.schema import ModelConfig


@dataclass(frozen=True)
class LoadedModel:
    """Loaded Hugging Face model bundle."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    name: str


def load_hf_causal_lm(config: ModelConfig) -> LoadedModel:
    """Load a causal language model and tokenizer from Hugging Face."""

    model_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.torch_dtype != "auto":
        model_kwargs["dtype"] = _parse_dtype(config.torch_dtype)
    else:
        model_kwargs["dtype"] = "auto"
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map

    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.name, **model_kwargs)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, name=config.name)


def _parse_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype}") from exc
