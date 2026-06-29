"""Pareto frontier plotting for sweep experiments."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast


def plot_pareto_frontier(
    sweep_results: dict[str, object],
    output_path: str | Path,
    *,
    title: str = "ComputeOS PVS: Latency vs Reference Perplexity",
) -> Path:
    """Plot a latency-vs-perplexity Pareto frontier from sweep results."""

    try:
        pyplot = cast(Any, import_module("matplotlib.pyplot"))
    except ImportError as exc:
        raise ImportError(
            "plot_pareto_frontier() requires matplotlib. "
            "Install it with: pip install 'computeos[visualization]'"
        ) from exc

    conditions = sweep_results.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("sweep_results must contain a 'conditions' mapping.")

    points = _condition_points(conditions)
    if not points:
        raise ValueError("sweep_results contains no plottable conditions.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = pyplot.subplots(figsize=(8, 5.5))
    pvs_points = [point for point in points if point[0] != "baseline"]
    if pvs_points:
        reductions = [point[3] for point in pvs_points]
        scatter = ax.scatter(
            [point[1] for point in pvs_points],
            [point[2] for point in pvs_points],
            c=reductions,
            cmap="Blues",
            s=90,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )
        fig.colorbar(scatter, ax=ax, label="Latency reduction (%)")

    for name, latency, perplexity, _reduction in points:
        if name == "baseline":
            ax.scatter(
                [latency],
                [perplexity],
                marker="*",
                s=200,
                color="grey",
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
        ax.annotate(name, (latency, perplexity), textcoords="offset points", xytext=(6, 6))

    frontier = _pareto_frontier(points)
    if len(frontier) >= 2:
        ax.plot(
            [point[1] for point in frontier],
            [point[2] for point in frontier],
            linestyle="--",
            color="black",
            linewidth=1.2,
            label="Pareto frontier",
            zorder=3,
        )
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("Mean generation latency (ms)")
    ax.set_ylabel("Reference perplexity")
    ax.grid(True, alpha=0.25)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    pyplot.close(fig)
    return output


def _condition_points(conditions: dict[object, object]) -> list[tuple[str, float, float, float]]:
    points: list[tuple[str, float, float, float]] = []
    for name, raw in conditions.items():
        if not isinstance(raw, dict):
            continue
        points.append(
            (
                str(name),
                float(raw.get("mean_latency_ms", 0.0)),
                float(raw.get("mean_perplexity", 0.0)),
                float(raw.get("latency_reduction_pct", 0.0)),
            )
        )
    return points


def _pareto_frontier(
    points: list[tuple[str, float, float, float]],
) -> list[tuple[str, float, float, float]]:
    frontier: list[tuple[str, float, float, float]] = []
    best_perplexity = float("inf")
    for point in sorted(points, key=lambda item: item[1]):
        if point[2] <= best_perplexity:
            frontier.append(point)
            best_perplexity = point[2]
    return frontier
