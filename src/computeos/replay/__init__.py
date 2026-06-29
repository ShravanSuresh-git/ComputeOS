"""Counterfactual Runtime Intelligence for ComputeOS."""

from computeos.replay.counterfactual_engine import CounterfactualEngine, CounterfactualResult
from computeos.replay.metrics import ReplayMetrics
from computeos.replay.oracle_scheduler import OracleObjective, OracleScheduler
from computeos.replay.regret import aggregate_batch_regret
from computeos.replay.scenario import CounterfactualScenario, ScenarioType
from computeos.replay.trace_loader import ReplayTrace, RuntimeEvent, TraceLoader
from computeos.replay.trace_player import ReplayState, TracePlayer

__all__ = [
    "CounterfactualEngine",
    "CounterfactualResult",
    "CounterfactualScenario",
    "OracleObjective",
    "OracleScheduler",
    "ReplayMetrics",
    "ReplayState",
    "ReplayTrace",
    "RuntimeEvent",
    "ScenarioType",
    "TraceLoader",
    "TracePlayer",
    "aggregate_batch_regret",
]
