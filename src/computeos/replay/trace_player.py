"""Deterministic replay player for completed runtime traces."""

from __future__ import annotations

from dataclasses import dataclass

from computeos.replay.trace_loader import ReplayTrace, RuntimeEvent


@dataclass(frozen=True)
class ReplayState:
    """Current replay cursor state."""

    position: int
    paused: bool
    speed: float
    event: RuntimeEvent | None


class TracePlayer:
    """Step, seek, pause, and resume deterministic runtime traces."""

    def __init__(self, trace: ReplayTrace, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("Replay speed must be positive.")
        self._trace = trace
        self._position = 0
        self._paused = True
        self._speed = speed

    @property
    def state(self) -> ReplayState:
        event = self._trace.events[self._position] if self._trace.events else None
        return ReplayState(
            position=self._position,
            paused=self._paused,
            speed=self._speed,
            event=event,
        )

    def pause(self) -> ReplayState:
        self._paused = True
        return self.state

    def resume(self) -> ReplayState:
        self._paused = False
        return self.state

    def step(self, count: int = 1) -> ReplayState:
        if count < 0:
            raise ValueError("Step count must be non-negative.")
        if self._trace.events:
            self._position = min(len(self._trace.events) - 1, self._position + count)
        return self.state

    def seek(self, position: int) -> ReplayState:
        if not self._trace.events:
            self._position = 0
            return self.state
        self._position = min(max(0, position), len(self._trace.events) - 1)
        return self.state

    def set_speed(self, speed: float) -> ReplayState:
        if speed <= 0:
            raise ValueError("Replay speed must be positive.")
        self._speed = speed
        return self.state

    def iter_events(self) -> tuple[RuntimeEvent, ...]:
        """Return the deterministic event sequence for non-interactive replay."""

        return self._trace.events
