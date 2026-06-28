"""Benchmark-style CRI experiment helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from html import escape
import json
from pathlib import Path

from computeos.replay.counterfactual_engine import CounterfactualEngine, CounterfactualResult
from computeos.replay.oracle_scheduler import OracleObjective, OracleScheduler
from computeos.replay.scenario import CounterfactualScenario, ScenarioType
from computeos.replay.trace_loader import ReplayTrace


@dataclass(frozen=True)
class PolicyComparison:
    """One row in a CRI policy comparison table."""

    policy: str
    utility: float
    latency_ms: float
    compute_units: float
    memory_mb: float
    budget_efficiency: float
    oracle_gap: float
    normalized_regret: float


class CounterfactualExperiment:
    """Run offline policy comparisons over a completed trace."""

    def __init__(self, engine: CounterfactualEngine | None = None) -> None:
        self._engine = engine or CounterfactualEngine()
        self._oracle = OracleScheduler()

    def default_policy_comparison(self, trace: ReplayTrace) -> list[PolicyComparison]:
        scenarios = [
            CounterfactualScenario(
                "static",
                ScenarioType.REPLACE_SCHEDULER,
                scheduler_name="static",
            ),
            CounterfactualScenario(
                "entropy",
                ScenarioType.REPLACE_SCHEDULER,
                scheduler_name="entropy",
            ),
            CounterfactualScenario(
                "confidence",
                ScenarioType.REPLACE_SCHEDULER,
                scheduler_name="confidence",
            ),
            CounterfactualScenario(
                "random",
                ScenarioType.REPLACE_SCHEDULER,
                scheduler_name="random",
            ),
            CounterfactualScenario("pvs", ScenarioType.REPLACE_SCHEDULER, scheduler_name="pvs"),
        ]
        results = self._engine.evaluate_many(trace, scenarios)
        rows = [_row_from_result(result) for result in results]
        oracle = self._oracle.plan(trace, objective=OracleObjective.MAXIMIZE_UTILITY)
        rows.append(
            PolicyComparison(
                policy="oracle",
                utility=oracle.utility,
                latency_ms=trace.total_latency_ms,
                compute_units=trace.total_compute_units,
                memory_mb=trace.peak_memory_mb,
                budget_efficiency=oracle.utility
                / max(1e-9, trace.total_latency_ms + trace.total_compute_units),
                oracle_gap=0.0,
                normalized_regret=0.0,
            )
        )
        return rows

    def export(
        self,
        rows: list[PolicyComparison],
        output_dir: str | Path,
        stem: str = "cri_policy_comparison",
    ) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": output / f"{stem}.json",
            "csv": output / f"{stem}.csv",
            "markdown": output / f"{stem}.md",
            "latex": output / f"{stem}.tex",
            "html": output / f"{stem}.html",
        }
        paths["json"].write_text(
            json.dumps([asdict(row) for row in rows], indent=2),
            encoding="utf-8",
        )
        with paths["csv"].open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[field.name for field in fields(PolicyComparison)],
            )
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        paths["markdown"].write_text(to_markdown(rows), encoding="utf-8")
        paths["latex"].write_text(to_latex(rows), encoding="utf-8")
        paths["html"].write_text(to_html(rows), encoding="utf-8")
        return paths


def to_markdown(rows: list[PolicyComparison]) -> str:
    headers = [field.name for field in fields(PolicyComparison)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [_format_markdown(value) for value in asdict(row).values()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def to_latex(rows: list[PolicyComparison]) -> str:
    headers = [field.name for field in fields(PolicyComparison)]
    lines = [
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        " \\toprule",
        " & ".join(headers) + " \\\\",
        " \\midrule",
    ]
    for row in rows:
        values = [_format_latex(value) for value in asdict(row).values()]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend([" \\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def to_html(rows: list[PolicyComparison]) -> str:
    headers = [field.name for field in fields(PolicyComparison)]
    lines = ["<table>", "  <thead><tr>"]
    lines.extend(f"    <th>{escape(header)}</th>" for header in headers)
    lines.extend(["  </tr></thead>", "  <tbody>"])
    for row in rows:
        lines.append("    <tr>")
        lines.extend(f"      <td>{escape(_format(value))}</td>" for value in asdict(row).values())
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>", ""])
    return "\n".join(lines)


def _row_from_result(result: CounterfactualResult) -> PolicyComparison:
    return PolicyComparison(
        policy=result.scenario.name,
        utility=result.predicted_utility,
        latency_ms=result.predicted_latency_ms,
        compute_units=result.predicted_compute_units,
        memory_mb=result.predicted_memory_mb,
        budget_efficiency=result.metrics.budget_efficiency,
        oracle_gap=result.metrics.oracle_gap,
        normalized_regret=result.regret.normalized_regret,
    )


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
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
