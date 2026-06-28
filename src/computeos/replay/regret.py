"""Scheduler regret metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerRegret:
    """Regret summary comparing an online policy to an oracle."""

    token_regret: tuple[float, ...]
    sequence_regret: float
    batch_regret: float
    average_regret: float
    normalized_regret: float


def compute_regret(
    oracle_utilities: list[float],
    online_utilities: list[float],
) -> SchedulerRegret:
    """Compute token, sequence, batch, average, and normalized regret."""

    length = max(len(oracle_utilities), len(online_utilities))
    if length == 0:
        return SchedulerRegret((), 0.0, 0.0, 0.0, 0.0)

    regrets: list[float] = []
    for index in range(length):
        oracle = oracle_utilities[index] if index < len(oracle_utilities) else oracle_utilities[-1]
        online = online_utilities[index] if index < len(online_utilities) else online_utilities[-1]
        regrets.append(max(0.0, oracle - online))

    sequence_regret = sum(regrets)
    oracle_total = sum(oracle_utilities) if oracle_utilities else 0.0
    normalized = sequence_regret / max(abs(oracle_total), 1e-9)
    return SchedulerRegret(
        token_regret=tuple(regrets),
        sequence_regret=sequence_regret,
        batch_regret=sequence_regret,
        average_regret=sequence_regret / length,
        normalized_regret=normalized,
    )
