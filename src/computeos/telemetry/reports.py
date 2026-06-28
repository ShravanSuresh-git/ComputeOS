"""Human-readable telemetry reports."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.table import Table

from computeos.telemetry.metrics import ModelTelemetry


def render_telemetry_report(telemetry: ModelTelemetry, max_layers: int = 16) -> str:
    """Render a compact terminal report for one inference run."""

    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    console.print(_summary_table(telemetry))
    console.print(_layer_table(telemetry, max_layers=max_layers))
    return output.getvalue().rstrip()


def print_telemetry_report(telemetry: ModelTelemetry, max_layers: int = 16) -> None:
    """Print a compact terminal report for one inference run."""

    console = Console()
    console.print(_summary_table(telemetry))
    console.print(_layer_table(telemetry, max_layers=max_layers))


def _summary_table(telemetry: ModelTelemetry) -> object:
    table = Table(title="ComputeOS Telemetry Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Model", telemetry.model_name)
    table.add_row("Total latency", _format_ms(telemetry.total_latency_ms))
    table.add_row("Layer events", str(len(telemetry.layers)))
    table.add_row("Scheduler decisions", str(len(telemetry.scheduler_decisions)))
    table.add_row("Token confidence mean", _format_float(_mean(telemetry.confidence_scores)))
    table.add_row(
        "Scheduler confidence mean",
        _format_float(_mean(telemetry.scheduler_confidence_scores)),
    )
    table.add_row("Peak CUDA memory", _format_bytes(telemetry.peak_memory_bytes))
    table.add_row("Peak RSS", _format_bytes(telemetry.peak_process_rss_bytes))
    return table


def _layer_table(telemetry: ModelTelemetry, max_layers: int) -> object:
    table = Table(title=f"Layer Events First {min(max_layers, len(telemetry.layers))}")
    table.add_column("#", justify="right")
    table.add_column("Layer")
    table.add_column("Type")
    table.add_column("Latency", justify="right")
    table.add_column("Entropy", justify="right")
    table.add_column("RSS", justify="right")
    table.add_column("Activation mean", justify="right")

    for index, layer in enumerate(telemetry.layers[:max_layers]):
        table.add_row(
            str(index),
            layer.layer_name,
            layer.layer_type,
            _format_ms(layer.latency_ms),
            _format_float(layer.attention_entropy),
            _format_bytes(layer.process_rss_bytes),
            _format_float(None if layer.activation_stats is None else layer.activation_stats.mean),
        )
    return table


def _plain_report(telemetry: ModelTelemetry, max_layers: int) -> str:
    lines = [
        "ComputeOS Telemetry Summary",
        f"Model: {telemetry.model_name}",
        f"Total latency: {_format_ms(telemetry.total_latency_ms)}",
        f"Layer events: {len(telemetry.layers)}",
        f"Scheduler decisions: {len(telemetry.scheduler_decisions)}",
        f"Token confidence mean: {_format_float(_mean(telemetry.confidence_scores))}",
        f"Scheduler confidence mean: {_format_float(_mean(telemetry.scheduler_confidence_scores))}",
        f"Peak CUDA memory: {_format_bytes(telemetry.peak_memory_bytes)}",
        f"Peak RSS: {_format_bytes(telemetry.peak_process_rss_bytes)}",
        "",
        f"Layer Events First {min(max_layers, len(telemetry.layers))}",
        "index | layer | type | latency | entropy | rss | activation_mean",
    ]
    for index, layer in enumerate(telemetry.layers[:max_layers]):
        activation_mean = None if layer.activation_stats is None else layer.activation_stats.mean
        lines.append(
            " | ".join(
                (
                    str(index),
                    layer.layer_name,
                    layer.layer_type,
                    _format_ms(layer.latency_ms),
                    _format_float(layer.attention_entropy),
                    _format_bytes(layer.process_rss_bytes),
                    _format_float(activation_mean),
                )
            )
        )
    return "\n".join(lines)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} ms"


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    units = ("B", "KB", "MB", "GB")
    scaled = float(value)
    unit = units[0]
    for unit in units:
        if scaled < 1024.0 or unit == units[-1]:
            break
        scaled /= 1024.0
    return f"{scaled:.2f} {unit}"
