"""Predictive Value Scheduling.

PVS treats adaptive inference as an optimal stopping problem. At each runtime
decision point it estimates the marginal value of spending additional compute
and requests early stopping when expected net value is no longer positive under
the configured resource budgets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, log1p

from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.telemetry.metrics import LayerTelemetry


@dataclass(frozen=True)
class PVSResourceBudgets:
    """Resource constraints used by Predictive Value Scheduling."""

    max_latency_ms: float = 250.0
    max_memory_mb: float = 4096.0
    max_compute_units: float = 512.0
    min_net_value: float = 0.0


@dataclass(frozen=True)
class PVSValueWeights:
    """Linear coefficients for the lightweight expected-utility model."""

    entropy: float = 0.25
    uncertainty: float = 0.30
    activation_norm: float = 0.15
    layer_variance: float = 0.15
    attention_entropy: float = 0.10
    decision_pressure: float = 0.05


@dataclass(frozen=True)
class PVSCostWeights:
    """Cost coefficients for compute, latency, and memory."""

    compute: float = 0.45
    latency: float = 0.35
    memory: float = 0.20


@dataclass(frozen=True)
class PVSFeatureVector:
    """Normalized runtime features consumed by the PVS prediction model."""

    entropy: float
    uncertainty: float
    activation_norm: float
    layer_variance: float
    attention_entropy: float
    decision_pressure: float
    normalized_latency: float
    normalized_memory: float
    compute_fraction: float


@dataclass(frozen=True)
class PVSPrediction:
    """Expected utility and resource cost estimates for one decision point."""

    expected_improvement: float
    expected_compute_cost: float
    expected_latency_ms: float
    expected_memory_mb: float
    expected_utility: float
    expected_cost: float
    expected_net_value: float
    stop_probability: float


@dataclass(frozen=True)
class PVSTraceEvent:
    """Replayable PVS decision trace event."""

    step_index: int
    layer_name: str | None
    action: str
    reason: str
    features: PVSFeatureVector
    prediction: PVSPrediction
    cumulative_latency_ms: float
    cumulative_compute_units: float
    peak_memory_mb: float
    remaining_latency_budget_ms: float
    remaining_compute_budget: float


@dataclass
class PredictiveValueScheduler(Scheduler):
    """Optimal-stopping scheduler based on predicted marginal value.

    The scheduler is intentionally pure: it does not execute inference. It emits
    immutable `SchedulerDecision` objects whose metadata contains replayable PVS
    features and predictions. Current ComputeOS records those decisions; future
    runtimes can apply `EARLY_EXIT` when backend capabilities allow it.
    """

    budgets: PVSResourceBudgets = field(default_factory=PVSResourceBudgets)
    value_weights: PVSValueWeights = field(default_factory=PVSValueWeights)
    cost_weights: PVSCostWeights = field(default_factory=PVSCostWeights)
    utility_scale: float = 1.0
    cost_scale: float = 1.0
    smoothing: float = 0.2
    cumulative_latency_ms: float = 0.0
    cumulative_compute_units: float = 0.0
    peak_memory_mb: float = 0.0
    stopped: bool = False
    _trace: list[PVSTraceEvent] = field(default_factory=list)
    _previous_activation_norm: float | None = None

    def reset(self) -> None:
        self.cumulative_latency_ms = 0.0
        self.cumulative_compute_units = 0.0
        self.peak_memory_mb = 0.0
        self.stopped = False
        self._trace.clear()
        self._previous_activation_norm = None

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        layer = context.layer_telemetry
        if layer is None:
            return SchedulerDecision.record_only("pvs waiting for layer telemetry")

        self._update_resource_state(layer)
        features = self._features(context, layer)
        prediction = self._predict(features)
        action, reason = self._select_action(prediction)

        event = PVSTraceEvent(
            step_index=context.step_index,
            layer_name=layer.layer_name,
            action=str(action),
            reason=reason,
            features=features,
            prediction=prediction,
            cumulative_latency_ms=self.cumulative_latency_ms,
            cumulative_compute_units=self.cumulative_compute_units,
            peak_memory_mb=self.peak_memory_mb,
            remaining_latency_budget_ms=self._remaining_latency_budget(),
            remaining_compute_budget=self._remaining_compute_budget(),
        )
        self._trace.append(event)

        if action == SchedulerAction.EARLY_EXIT:
            self.stopped = True

        return SchedulerDecision(
            action=action,
            layer_name=layer.layer_name,
            confidence=1.0 - prediction.stop_probability
            if action == SchedulerAction.CONTINUE
            else prediction.stop_probability,
            reason=reason,
            metadata={
                "algorithm": "predictive_value_scheduling",
                "features": asdict(features),
                "prediction": asdict(prediction),
                "budgets": asdict(self.budgets),
                "cumulative_latency_ms": self.cumulative_latency_ms,
                "cumulative_compute_units": self.cumulative_compute_units,
                "peak_memory_mb": self.peak_memory_mb,
                "remaining_latency_budget_ms": self._remaining_latency_budget(),
                "remaining_compute_budget": self._remaining_compute_budget(),
                "stopping_event": action == SchedulerAction.EARLY_EXIT,
            },
        )

    def observe(self, context: SchedulerContext, decision: SchedulerDecision) -> None:
        """PVS is deterministic in this implementation; observations are logged via metadata."""

    def replay(self) -> tuple[PVSTraceEvent, ...]:
        """Return an immutable replay trace for the current request."""

        return tuple(self._trace)

    def _update_resource_state(self, layer: LayerTelemetry) -> None:
        self.cumulative_latency_ms += max(0.0, layer.latency_ms)
        self.cumulative_compute_units += _compute_units(layer)
        memory_mb = _memory_mb(layer)
        if memory_mb is not None:
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

    def _features(self, context: SchedulerContext, layer: LayerTelemetry) -> PVSFeatureVector:
        activation_norm = _activation_norm(layer)
        layer_variance = _layer_variance(
            activation_norm=activation_norm,
            previous_activation_norm=self._previous_activation_norm,
        )
        self._previous_activation_norm = activation_norm

        confidence = _latest_confidence(context)
        uncertainty = 1.0 - confidence
        entropy = _confidence_entropy(confidence)
        attention = _normalize(layer.attention_entropy, scale=4.0)
        decision_pressure = _normalize(len(context.model_telemetry.scheduler_decisions), scale=64.0)

        return PVSFeatureVector(
            entropy=entropy,
            uncertainty=uncertainty,
            activation_norm=activation_norm,
            layer_variance=layer_variance,
            attention_entropy=attention,
            decision_pressure=decision_pressure,
            normalized_latency=_ratio(self.cumulative_latency_ms, self.budgets.max_latency_ms),
            normalized_memory=_ratio(self.peak_memory_mb, self.budgets.max_memory_mb),
            compute_fraction=_ratio(
                self.cumulative_compute_units,
                self.budgets.max_compute_units,
            ),
        )

    def _predict(self, features: PVSFeatureVector) -> PVSPrediction:
        expected_improvement = _clamp01(
            self.value_weights.entropy * features.entropy
            + self.value_weights.uncertainty * features.uncertainty
            + self.value_weights.activation_norm * features.activation_norm
            + self.value_weights.layer_variance * features.layer_variance
            + self.value_weights.attention_entropy * features.attention_entropy
            + self.value_weights.decision_pressure * features.decision_pressure
        )
        expected_compute_cost = _clamp01(features.compute_fraction + 0.05)
        expected_latency_ms = self._expected_next_latency_ms(features)
        expected_memory_mb = max(
            self.peak_memory_mb,
            features.normalized_memory * self.budgets.max_memory_mb,
        )

        normalized_latency_cost = _ratio(expected_latency_ms, self.budgets.max_latency_ms)
        normalized_memory_cost = _ratio(expected_memory_mb, self.budgets.max_memory_mb)
        expected_cost = _clamp01(
            self.cost_weights.compute * expected_compute_cost
            + self.cost_weights.latency * normalized_latency_cost
            + self.cost_weights.memory * normalized_memory_cost
        )
        expected_utility = self.utility_scale * expected_improvement
        scaled_cost = self.cost_scale * expected_cost
        expected_net_value = expected_utility - scaled_cost
        stop_probability = _sigmoid(-expected_net_value)

        return PVSPrediction(
            expected_improvement=expected_improvement,
            expected_compute_cost=expected_compute_cost,
            expected_latency_ms=expected_latency_ms,
            expected_memory_mb=expected_memory_mb,
            expected_utility=expected_utility,
            expected_cost=scaled_cost,
            expected_net_value=expected_net_value,
            stop_probability=stop_probability,
        )

    def _select_action(self, prediction: PVSPrediction) -> tuple[SchedulerAction, str]:
        if self.stopped:
            return SchedulerAction.RECORD_ONLY, "pvs already emitted a stopping decision"
        if self.cumulative_latency_ms >= self.budgets.max_latency_ms:
            return SchedulerAction.EARLY_EXIT, "latency budget exhausted"
        if self.cumulative_compute_units >= self.budgets.max_compute_units:
            return SchedulerAction.EARLY_EXIT, "compute budget exhausted"
        if self.peak_memory_mb >= self.budgets.max_memory_mb:
            return SchedulerAction.EARLY_EXIT, "memory budget exhausted"
        if prediction.expected_net_value <= self.budgets.min_net_value:
            return SchedulerAction.EARLY_EXIT, "expected net value below stopping threshold"
        return SchedulerAction.CONTINUE, "expected marginal value exceeds estimated cost"

    def _expected_next_latency_ms(self, features: PVSFeatureVector) -> float:
        pressure = 1.0 + self.smoothing * features.decision_pressure
        return max(0.0, self.cumulative_latency_ms * pressure)

    def _remaining_latency_budget(self) -> float:
        return max(0.0, self.budgets.max_latency_ms - self.cumulative_latency_ms)

    def _remaining_compute_budget(self) -> float:
        return max(0.0, self.budgets.max_compute_units - self.cumulative_compute_units)


def _latest_confidence(context: SchedulerContext) -> float:
    scores = context.model_telemetry.confidence_scores
    if not scores:
        return 0.5
    return _clamp01(scores[-1])


def _confidence_entropy(confidence: float) -> float:
    confidence = min(max(confidence, 1e-6), 1.0 - 1e-6)
    entropy = -(confidence * log1p(confidence - 1.0) + (1.0 - confidence) * log1p(-confidence))
    return _clamp01(entropy / 0.6931471805599453)


def _activation_norm(layer: LayerTelemetry) -> float:
    if layer.activation_stats is None:
        return 0.0
    return _clamp01(log1p(abs(layer.activation_stats.l2_norm)) / 8.0)


def _layer_variance(activation_norm: float, previous_activation_norm: float | None) -> float:
    if previous_activation_norm is None:
        return 0.0
    return _clamp01(abs(activation_norm - previous_activation_norm))


def _compute_units(layer: LayerTelemetry) -> float:
    if layer.activation_stats is None:
        return 1.0
    return max(1.0, layer.activation_stats.numel / 1024.0)


def _memory_mb(layer: LayerTelemetry) -> float | None:
    value = layer.memory_allocated_bytes or layer.process_rss_bytes
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def _normalize(value: float | None, scale: float) -> float:
    if value is None:
        return 0.0
    return _clamp01(value / scale)


def _ratio(value: float, budget: float) -> float:
    if budget <= 0:
        return 1.0
    return _clamp01(value / budget)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)
