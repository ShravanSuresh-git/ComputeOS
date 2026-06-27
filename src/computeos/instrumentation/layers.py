"""Transformer layer discovery utilities."""

from __future__ import annotations

from collections.abc import Iterable
import re

import torch.nn as nn

_TRANSFORMER_LAYER_HINTS = (
    "BertLayer",
    "GPT2Block",
    "LlamaDecoderLayer",
    "MistralDecoderLayer",
    "OPTDecoderLayer",
    "T5Block",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
)


def discover_transformer_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find likely transformer blocks in a Hugging Face or PyTorch model."""

    matches: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if _is_transformer_layer(name, module):
            matches.append((name, module))
    return matches


def _is_transformer_layer(name: str, module: nn.Module) -> bool:
    class_name = module.__class__.__name__
    if class_name in _TRANSFORMER_LAYER_HINTS:
        return True
    return any(
        re.fullmatch(pattern, name)
        for pattern in (
            r"transformer\.h\.\d+",
            r"model\.layers\.\d+",
            r"encoder\.layer\.\d+",
            r"decoder\.block\.\d+",
        )
    )


def module_names(modules: Iterable[tuple[str, nn.Module]]) -> tuple[str, ...]:
    return tuple(name for name, _ in modules)
