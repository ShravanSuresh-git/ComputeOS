"""Telemetry statistic helpers."""

from __future__ import annotations

import torch

from computeos.telemetry.metrics import ActivationStats


def first_tensor(value: object) -> torch.Tensor | None:
    """Return the first tensor inside common PyTorch output structures."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def activation_stats(value: object) -> ActivationStats | None:
    """Compute stable activation summary statistics without retaining tensors."""

    tensor = first_tensor(value)
    if tensor is None or tensor.numel() == 0:
        return None

    detached = tensor.detach().float()
    return ActivationStats(
        mean=float(detached.mean().item()),
        std=float(detached.std(unbiased=False).item()),
        min=float(detached.min().item()),
        max=float(detached.max().item()),
        l2_norm=float(torch.linalg.vector_norm(detached).item()),
        numel=int(detached.numel()),
    )


def attention_entropy(value: object) -> float | None:
    """Estimate mean attention entropy when attention probabilities are present."""

    tensor = _find_attention_tensor(value)
    if tensor is None or tensor.numel() == 0:
        return None

    probs = tensor.detach().float().clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return float(entropy.item())


def _find_attention_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor) and value.ndim >= 3:
        last_dim_sum = value.detach().float().sum(dim=-1).mean()
        if torch.isfinite(last_dim_sum) and abs(float(last_dim_sum.item()) - 1.0) < 0.1:
            return value
    if isinstance(value, dict):
        for key in ("attentions", "attention_probs", "attn_weights"):
            if key in value:
                found = first_tensor(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = _find_attention_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _find_attention_tensor(item)
            if found is not None:
                return found
    return None
