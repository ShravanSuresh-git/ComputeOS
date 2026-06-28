"""Publication-style benchmark report exports."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from html import escape
import json
from pathlib import Path

from computeos.benchmarks.base import BenchmarkResult


@dataclass(frozen=True)
class BenchmarkReportRow:
    """Flat benchmark result row suitable for publication exports."""

    prompt: str
    generated_text: str
    score: float | None
    total_latency_ms: float | None
    layer_events: int
    scheduler_decisions: int
    peak_memory_bytes: int | None
    peak_process_rss_bytes: int | None


def rows_from_results(results: list[BenchmarkResult]) -> list[BenchmarkReportRow]:
    """Convert benchmark results into report rows."""

    rows: list[BenchmarkReportRow] = []
    for result in results:
        telemetry = result.execution.telemetry
        rows.append(
            BenchmarkReportRow(
                prompt=result.item.prompt,
                generated_text=result.execution.generated_text,
                score=result.score,
                total_latency_ms=telemetry.total_latency_ms,
                layer_events=len(telemetry.layers),
                scheduler_decisions=len(telemetry.scheduler_decisions),
                peak_memory_bytes=telemetry.peak_memory_bytes,
                peak_process_rss_bytes=telemetry.peak_process_rss_bytes,
            )
        )
    return rows


def export_benchmark_report(
    rows: list[BenchmarkReportRow],
    output_dir: str | Path,
    stem: str = "benchmark_report",
) -> dict[str, Path]:
    """Export benchmark rows as CSV, JSON, Markdown, LaTeX, and HTML."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output / f"{stem}.csv",
        "json": output / f"{stem}.json",
        "markdown": output / f"{stem}.md",
        "latex": output / f"{stem}.tex",
        "html": output / f"{stem}.html",
    }
    payload = [asdict(row) for row in rows]
    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_benchmark_report_fieldnames())
        writer.writeheader()
        writer.writerows(payload)
    paths["markdown"].write_text(to_markdown(rows), encoding="utf-8")
    paths["latex"].write_text(to_latex(rows), encoding="utf-8")
    paths["html"].write_text(to_html(rows), encoding="utf-8")
    return paths


def to_markdown(rows: list[BenchmarkReportRow]) -> str:
    payload = [asdict(row) for row in rows]
    headers = list(payload[0].keys()) if payload else _benchmark_report_fieldnames()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in payload:
        lines.append("| " + " | ".join(_format_markdown(value) for value in row.values()) + " |")
    return "\n".join(lines) + "\n"


def to_latex(rows: list[BenchmarkReportRow]) -> str:
    payload = [asdict(row) for row in rows]
    headers = list(payload[0].keys()) if payload else _benchmark_report_fieldnames()
    lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        " \\toprule",
        " & ".join(headers) + " \\\\",
        " \\midrule",
    ]
    for row in payload:
        lines.append(" & ".join(_format_latex(value) for value in row.values()) + " \\\\")
    lines.extend([" \\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def to_html(rows: list[BenchmarkReportRow]) -> str:
    payload = [asdict(row) for row in rows]
    headers = list(payload[0].keys()) if payload else _benchmark_report_fieldnames()
    lines = ["<table>", "  <thead><tr>"]
    lines.extend(f"    <th>{escape(header)}</th>" for header in headers)
    lines.extend(["  </tr></thead>", "  <tbody>"])
    for row in payload:
        lines.append("    <tr>")
        lines.extend(f"      <td>{escape(_format(value))}</td>" for value in row.values())
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>", ""])
    return "\n".join(lines)


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def _format_markdown(value: object) -> str:
    return _format(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _format_latex(value: object) -> str:
    text = _format(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _benchmark_report_fieldnames() -> list[str]:
    return [field.name for field in fields(BenchmarkReportRow)]
