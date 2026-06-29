"""Pareto frontier analysis for quality/latency tradeoff reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParetoPoint:
    """One point on the quality/latency Pareto frontier."""

    scheduler: str
    latency_ms: float
    score: float
    row: dict[str, Any]


def pareto_frontier(rows: list[dict[str, Any]]) -> list[ParetoPoint]:
    """Return rows not dominated on latency and score.

    A point A dominates point B if A has lower latency and lower perplexity.
    Rows missing `latency_ms` or `score` are skipped.
    """

    points: list[ParetoPoint] = []
    for row in rows:
        latency = row.get("latency_ms")
        score = row.get("score")
        if latency is None or score is None:
            continue
        try:
            points.append(
                ParetoPoint(
                    scheduler=str(row.get("scheduler", "unknown")),
                    latency_ms=float(latency),
                    score=float(score),
                    row=row,
                )
            )
        except (TypeError, ValueError):
            continue

    frontier: list[ParetoPoint] = []
    for candidate in sorted(points, key=lambda point: point.latency_ms):
        dominated = any(
            other.latency_ms <= candidate.latency_ms
            and other.score <= candidate.score
            and (
                other.latency_ms < candidate.latency_ms
                or other.score < candidate.score
            )
            for other in points
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def plot_pareto(
    frontier: list[ParetoPoint],
    all_points: list[ParetoPoint],
    output_path: str | Path,
) -> Path:
    """Plot quality/latency Pareto frontier. Requires matplotlib."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_pareto() requires matplotlib. "
            "Install it with: pip install 'computeos[visualization]'"
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frontier_names = {point.scheduler for point in frontier}
    fig, axis = plt.subplots(figsize=(8, 5))

    for point in all_points:
        on_frontier = point.scheduler in frontier_names
        axis.scatter(
            point.latency_ms,
            point.score,
            marker="*" if on_frontier else "o",
            s=120 if on_frontier else 60,
            label=f"{point.scheduler} {'*' if on_frontier else ''}",
            zorder=3 if on_frontier else 2,
        )

    if frontier:
        sorted_frontier = sorted(frontier, key=lambda point: point.latency_ms)
        axis.plot(
            [point.latency_ms for point in sorted_frontier],
            [point.score for point in sorted_frontier],
            "k--",
            linewidth=1,
            alpha=0.4,
            label="Pareto frontier",
        )

    axis.set_xlabel("Latency (ms)")
    axis.set_ylabel("Perplexity (lower is better)")
    axis.set_title("Quality / Latency Pareto Frontier")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
