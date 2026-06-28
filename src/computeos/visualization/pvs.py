"""Visualization and replay helpers for Predictive Value Scheduling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from computeos.telemetry.metrics import ModelTelemetry


def extract_pvs_trace(telemetry: ModelTelemetry) -> list[dict[str, Any]]:
    """Extract replayable PVS records from scheduler decision metadata."""

    records: list[dict[str, Any]] = []
    for index, decision in enumerate(telemetry.scheduler_decisions):
        if decision.metadata.get("algorithm") != "predictive_value_scheduling":
            continue
        records.append(
            {
                "index": index,
                "action": str(decision.action),
                "layer_name": decision.layer_name,
                "reason": decision.reason,
                "features": decision.metadata.get("features", {}),
                "prediction": decision.metadata.get("prediction", {}),
                "cumulative_latency_ms": decision.metadata.get("cumulative_latency_ms"),
                "cumulative_compute_units": decision.metadata.get("cumulative_compute_units"),
                "peak_memory_mb": decision.metadata.get("peak_memory_mb"),
                "stopping_event": decision.metadata.get("stopping_event", False),
            }
        )
    return records


def plot_pvs_trace(telemetry: ModelTelemetry, output_path: str | Path) -> Path:
    """Generate a PVS timeline plot.

    The plot shows predicted improvement, expected cost, expected net value, and
    stopping events over scheduler decision time. Matplotlib is imported lazily
    so ComputeOS does not require plotting dependencies for normal runtime use.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "PVS visualization requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    trace = extract_pvs_trace(telemetry)
    if not trace:
        raise ValueError("No Predictive Value Scheduling records found in telemetry.")

    indices = [record["index"] for record in trace]
    predictions = [record["prediction"] for record in trace]
    expected_improvement = [
        float(prediction.get("expected_improvement", 0.0)) for prediction in predictions
    ]
    expected_cost = [float(prediction.get("expected_cost", 0.0)) for prediction in predictions]
    expected_net_value = [
        float(prediction.get("expected_net_value", 0.0)) for prediction in predictions
    ]
    stop_indices = [record["index"] for record in trace if record["stopping_event"]]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(indices, expected_improvement, label="Expected improvement")
    axis.plot(indices, expected_cost, label="Expected cost")
    axis.plot(indices, expected_net_value, label="Expected net value")
    for stop_index in stop_indices:
        axis.axvline(stop_index, color="red", linestyle="--", alpha=0.5)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axis.set_title("Predictive Value Scheduling Timeline")
    axis.set_xlabel("Scheduler decision index")
    axis.set_ylabel("Normalized value")
    axis.legend()
    axis.grid(True, alpha=0.25)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
