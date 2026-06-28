"""Visualization helpers for CRI replay traces."""

from __future__ import annotations

from pathlib import Path

from computeos.replay.counterfactual_engine import CounterfactualResult
from computeos.replay.trace_loader import ReplayTrace, RuntimeEventType


class ReplayVisualizer:
    """Generate publication-oriented replay figures."""

    def timeline(self, trace: ReplayTrace, output_path: str | Path) -> Path:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Replay visualization requires matplotlib.") from exc

        layer_events = [
            event for event in trace.events if event.event_type == RuntimeEventType.LAYER_FINISHED
        ]
        decision_events = [
            event
            for event in trace.events
            if event.event_type == RuntimeEventType.SCHEDULER_DECISION
        ]
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.scatter(
            [event.timestamp_offset_ms for event in layer_events],
            [event.index for event in layer_events],
            label="Layer execution",
        )
        axis.scatter(
            [event.timestamp_offset_ms for event in decision_events],
            [event.index for event in decision_events],
            label="Scheduler decision",
            marker="x",
        )
        axis.set_title("ComputeOS Replay Timeline")
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Event index")
        axis.legend()
        axis.grid(True, alpha=0.25)
        return _save(fig, output_path)

    def regret_timeline(
        self,
        result: CounterfactualResult,
        output_path: str | Path,
    ) -> Path:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Replay visualization requires matplotlib.") from exc

        regrets = list(result.regret.token_regret)
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.plot(range(len(regrets)), regrets, label="Scheduler regret")
        axis.set_title(f"Regret Timeline: {result.scenario.name}")
        axis.set_xlabel("Decision index")
        axis.set_ylabel("Regret")
        axis.legend()
        axis.grid(True, alpha=0.25)
        return _save(fig, output_path)


def _save(fig, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except ImportError:
        pass
    return path
