"""Experiment artifact snapshots."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from computeos.experiments.comparison import ComparisonReport
from computeos.telemetry.metrics import ModelTelemetry
from computeos.telemetry.serialization import flatten_telemetry_to_dict


class ArtifactStore:
    """Write reproducibility artifacts for one experiment run."""

    def __init__(self, output_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now(tz=UTC).isoformat()
        self.run_dir = output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def snapshot_config(self, cfg: dict[str, Any]) -> Path:
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
        return path

    def snapshot_env(self) -> Path:
        path = self.run_dir / "env.json"
        try:
            computeos_version = importlib.metadata.version("computeos")
        except importlib.metadata.PackageNotFoundError:
            computeos_version = "unknown"
        try:
            torch_version = str(torch.__version__)
        except Exception:
            torch_version = "unknown"
        try:
            import transformers

            transformers_version = transformers.__version__
        except Exception:
            transformers_version = "unknown"
        uname = platform.uname()
        env = {
            "python_version": sys.version,
            "torch_version": torch_version,
            "transformers_version": transformers_version,
            "computeos_version": computeos_version,
            "cuda_available": torch.cuda.is_available(),
            "platform": {
                "system": uname.system,
                "node": uname.node,
                "release": uname.release,
                "machine": uname.machine,
            },
        }
        path.write_text(json.dumps(env, indent=2), encoding="utf-8")
        return path

    def snapshot_telemetry(self, telemetry: ModelTelemetry) -> Path:
        path = self.run_dir / "telemetry.jsonl"
        record = flatten_telemetry_to_dict(telemetry)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=str) + "\n")
        return path

    def snapshot_report(self, report: ComparisonReport) -> Path:
        csv_path = self.run_dir / "report.csv"
        json_path = self.run_dir / "report.json"
        report.to_csv(csv_path)
        regret_data = {
            name: asdict(regret)
            for name, regret in report.regret_by_scheduler.items()
        }
        json_path.write_text(
            json.dumps({"regret": regret_data}, indent=2, default=str),
            encoding="utf-8",
        )
        return json_path
