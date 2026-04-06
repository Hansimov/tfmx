"""Adaptive routing telemetry and scoring helpers for QWN machine."""

import time

from collections import deque
from dataclasses import dataclass, field


RECENT_WINDOW_SEC = 60.0
DEFAULT_FAILURE_COOLDOWN_SEC = 8.0
DEFAULT_RECOVERY_COOLDOWN_SEC = 2.0
EMA_ALPHA = 0.25
LATENCY_FALLBACK_MS = 2500.0
TTFT_FALLBACK_MS = 1500.0
_NEUTRAL_METRIC_PENALTY = 0.35
_MAX_HISTORY = 256


@dataclass(frozen=True)
class SchedulerWeights:
    load: float = 0.32
    gpu_pressure: float = 0.24
    latency: float = 0.16
    ttft: float = 0.10
    throughput: float = 0.08
    failures: float = 0.08
    skew: float = 0.02

    @classmethod
    def components(cls) -> tuple[str, ...]:
        return (
            "load",
            "gpu_pressure",
            "latency",
            "ttft",
            "throughput",
            "failures",
            "skew",
        )

    def raw_dict(self) -> dict[str, float]:
        return {
            "load": self.load,
            "gpu_pressure": self.gpu_pressure,
            "latency": self.latency,
            "ttft": self.ttft,
            "throughput": self.throughput,
            "failures": self.failures,
            "skew": self.skew,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "SchedulerWeights":
        normalized = _normalize_weight_map(data)
        return cls(**normalized)

    def to_dict(self) -> dict[str, float]:
        return {
            "load": round(self.load, 4),
            "gpu_pressure": round(self.gpu_pressure, 4),
            "latency": round(self.latency, 4),
            "ttft": round(self.ttft, 4),
            "throughput": round(self.throughput, 4),
            "failures": round(self.failures, 4),
            "skew": round(self.skew, 4),
        }


@dataclass(frozen=True)
class SchedulerTuningConfig:
    enabled: bool = True
    target_latency_ms: float = 900.0
    target_ttft_ms: float = 300.0
    hot_gpu_pressure: float = 0.72
    saturation_active_ratio: float = 0.65
    throughput_cv_threshold: float = 0.08
    failure_rate_threshold: float = 0.03
    request_cv_threshold: float = 0.20

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "target_latency_ms": round(self.target_latency_ms, 2),
            "target_ttft_ms": round(self.target_ttft_ms, 2),
            "hot_gpu_pressure": round(self.hot_gpu_pressure, 4),
            "saturation_active_ratio": round(self.saturation_active_ratio, 4),
            "throughput_cv_threshold": round(self.throughput_cv_threshold, 4),
            "failure_rate_threshold": round(self.failure_rate_threshold, 4),
            "request_cv_threshold": round(self.request_cv_threshold, 4),
        }


@dataclass(frozen=True)
class SchedulerTuningResult:
    weights: SchedulerWeights
    signals: dict[str, float] = field(default_factory=dict)


def _new_history() -> deque[float]:
    return deque(maxlen=_MAX_HISTORY)


@dataclass
class InstanceTelemetry:
    recent_window_sec: float = RECENT_WINDOW_SEC
    ema_alpha: float = EMA_ALPHA
    latency_ema_ms: float = 0.0
    ttft_ema_ms: float = 0.0
    tokens_per_second_ema: float = 0.0
    total_successes: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    last_error_at: float = 0.0
    last_dispatch_at: float = 0.0
    last_completed_at: float = 0.0
    cooldown_until: float = 0.0
    dispatch_timestamps: deque[float] = field(default_factory=_new_history)
    success_timestamps: deque[float] = field(default_factory=_new_history)
    failure_timestamps: deque[float] = field(default_factory=_new_history)

    def _prune(self, now: float) -> None:
        cutoff = now - self.recent_window_sec
        for queue in (
            self.dispatch_timestamps,
            self.success_timestamps,
            self.failure_timestamps,
        ):
            while queue and queue[0] < cutoff:
                queue.popleft()

    def _update_ema(self, current: float, new_value: float) -> float:
        if new_value <= 0:
            return current
        if current <= 0:
            return new_value
        alpha = min(max(self.ema_alpha, 0.01), 1.0)
        return alpha * new_value + (1 - alpha) * current

    def record_dispatch(self, now: float | None = None) -> None:
        now = now or time.monotonic()
        self._prune(now)
        self.last_dispatch_at = now
        self.dispatch_timestamps.append(now)

    def record_success(
        self,
        latency_ms: float,
        ttft_ms: float = 0.0,
        completion_tokens: int = 0,
        now: float | None = None,
    ) -> None:
        now = now or time.monotonic()
        self._prune(now)
        self.total_successes += 1
        self.consecutive_failures = 0
        self.last_completed_at = now
        self.success_timestamps.append(now)
        self.latency_ema_ms = self._update_ema(self.latency_ema_ms, latency_ms)
        if ttft_ms > 0:
            self.ttft_ema_ms = self._update_ema(self.ttft_ema_ms, ttft_ms)
        if completion_tokens > 0 and latency_ms > 0:
            tokens_per_second = completion_tokens / (latency_ms / 1000.0)
            self.tokens_per_second_ema = self._update_ema(
                self.tokens_per_second_ema,
                tokens_per_second,
            )

    def record_failure(
        self,
        error: str = "",
        cooldown_sec: float = DEFAULT_FAILURE_COOLDOWN_SEC,
        now: float | None = None,
    ) -> None:
        now = now or time.monotonic()
        self._prune(now)
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_error = error or self.last_error
        self.last_error_at = now
        self.cooldown_until = max(self.cooldown_until, now + max(cooldown_sec, 0.0))
        self.failure_timestamps.append(now)

    def record_recovery(
        self,
        cooldown_sec: float = DEFAULT_RECOVERY_COOLDOWN_SEC,
        now: float | None = None,
    ) -> None:
        now = now or time.monotonic()
        self._prune(now)
        self.consecutive_failures = 0
        self.cooldown_until = max(self.cooldown_until, now + max(cooldown_sec, 0.0))

    def recent_requests(self, now: float | None = None) -> int:
        now = now or time.monotonic()
        self._prune(now)
        return len(self.dispatch_timestamps)

    def recent_successes(self, now: float | None = None) -> int:
        now = now or time.monotonic()
        self._prune(now)
        return len(self.success_timestamps)

    def recent_failures(self, now: float | None = None) -> int:
        now = now or time.monotonic()
        self._prune(now)
        return len(self.failure_timestamps)

    def recent_success_rate(self, now: float | None = None) -> float | None:
        requests = self.recent_requests(now)
        if requests <= 0:
            return None
        failures = self.recent_failures(now)
        return max(0.0, min(1.0, (requests - failures) / requests))

    def cooldown_remaining_sec(self, now: float | None = None) -> float:
        now = now or time.monotonic()
        return max(0.0, self.cooldown_until - now)

    def snapshot(self, now: float | None = None) -> dict:
        now = now or time.monotonic()
        recent_requests = self.recent_requests(now)
        recent_successes = self.recent_successes(now)
        recent_failures = self.recent_failures(now)
        success_rate = self.recent_success_rate(now)
        return {
            "recent_requests": recent_requests,
            "recent_successes": recent_successes,
            "recent_failures": recent_failures,
            "success_rate": (
                round(success_rate, 4) if success_rate is not None else None
            ),
            "latency_ema_ms": (
                round(self.latency_ema_ms, 2) if self.latency_ema_ms > 0 else None
            ),
            "ttft_ema_ms": round(self.ttft_ema_ms, 2) if self.ttft_ema_ms > 0 else None,
            "tokens_per_second_ema": (
                round(self.tokens_per_second_ema, 2)
                if self.tokens_per_second_ema > 0
                else None
            ),
            "cooldown_remaining_sec": round(self.cooldown_remaining_sec(now), 2),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
        }


def get_peer_baseline(values: list[float], fallback: float) -> float:
    positives = [value for value in values if value > 0]
    if not positives:
        return fallback
    return max(fallback * 0.5, sum(positives) / len(positives))


def _normalize_weight_map(values: dict[str, float]) -> dict[str, float]:
    components = SchedulerWeights.components()
    cleaned = {name: max(0.0, float(values.get(name, 0.0))) for name in components}
    total = sum(cleaned.values())
    if total <= 0:
        default_weights = SchedulerWeights().raw_dict()
        total = sum(default_weights.values())
        return {name: default_weights[name] / total for name in components}
    return {name: cleaned[name] / total for name in components}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _coefficient_of_variation(values: list[float]) -> float:
    positives = [value for value in values if value > 0]
    if len(positives) < 2:
        return 0.0
    mean_value = _mean(positives)
    if mean_value <= 0:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in positives) / len(positives)
    return (variance**0.5) / mean_value


def tune_scheduler_weights(
    *,
    base_weights: SchedulerWeights,
    active_ratios: list[float],
    gpu_pressures: list[float],
    latencies_ms: list[float],
    ttfts_ms: list[float],
    throughputs_tokens_per_second: list[float],
    recent_requests: list[int],
    failure_rates: list[float],
    consecutive_failures: list[int],
    config: SchedulerTuningConfig,
) -> SchedulerTuningResult:
    if not config.enabled:
        return SchedulerTuningResult(weights=base_weights, signals={})

    active_values = [max(0.0, value) for value in active_ratios]
    gpu_values = [min(max(value, 0.0), 1.0) for value in gpu_pressures]
    latency_values = [value for value in latencies_ms if value > 0]
    ttft_values = [value for value in ttfts_ms if value > 0]
    throughput_values = [value for value in throughputs_tokens_per_second if value > 0]
    request_values = [max(0, int(value)) for value in recent_requests]
    failure_values = [min(max(value, 0.0), 1.0) for value in failure_rates]
    consecutive_values = [max(0, int(value)) for value in consecutive_failures]

    avg_active_ratio = _mean(active_values)
    max_active_ratio = max(active_values) if active_values else 0.0
    avg_gpu_pressure = _mean(gpu_values)
    max_gpu_pressure = max(gpu_values) if gpu_values else 0.0
    gpu_pressure_cv = _coefficient_of_variation(gpu_values)
    avg_latency_ms = _mean(latency_values)
    max_latency_ms = max(latency_values) if latency_values else 0.0
    avg_ttft_ms = _mean(ttft_values)
    max_ttft_ms = max(ttft_values) if ttft_values else 0.0
    avg_throughput = _mean(throughput_values)
    min_throughput = min(throughput_values) if throughput_values else 0.0
    throughput_cv = _coefficient_of_variation(throughput_values)
    avg_failure_rate = _mean(failure_values)
    max_failure_rate = max(failure_values) if failure_values else 0.0
    max_consecutive_failures = max(consecutive_values) if consecutive_values else 0
    request_cv = _coefficient_of_variation([float(value) for value in request_values])

    active_saturation = max(0.0, avg_active_ratio - config.saturation_active_ratio)
    hot_gpu_excess = max(0.0, max_gpu_pressure - config.hot_gpu_pressure)
    latency_excess = max(0.0, avg_latency_ms / config.target_latency_ms - 1.0)
    ttft_excess = max(0.0, avg_ttft_ms / config.target_ttft_ms - 1.0)
    throughput_cv_excess = max(0.0, throughput_cv - config.throughput_cv_threshold)
    failure_excess = max(0.0, avg_failure_rate - config.failure_rate_threshold)
    request_cv_excess = max(0.0, request_cv - config.request_cv_threshold)

    throughput_floor_penalty = 0.0
    if avg_throughput > 0 and min_throughput > 0:
        throughput_floor_penalty = max(0.0, 1.0 - (min_throughput / avg_throughput))

    factors = {name: 1.0 for name in SchedulerWeights.components()}
    factors["load"] += active_saturation * 1.2 + max(0.0, max_active_ratio - 0.85) * 0.8
    factors["gpu_pressure"] += (
        max(0.0, avg_gpu_pressure - 0.45) * 0.8
        + hot_gpu_excess * 1.8
        + gpu_pressure_cv * 0.4
    )
    factors["latency"] += (
        latency_excess * 0.9
        + max(0.0, max_latency_ms / config.target_latency_ms - 1.4) * 0.5
    )
    factors["ttft"] += (
        ttft_excess * 1.0 + max(0.0, max_ttft_ms / config.target_ttft_ms - 1.5) * 0.4
    )
    factors["throughput"] += throughput_cv_excess * 1.2 + throughput_floor_penalty * 0.6
    factors["failures"] += (
        failure_excess * 2.5
        + max(0.0, max_failure_rate - config.failure_rate_threshold) * 3.0
        + max_consecutive_failures * 0.18
    )
    factors["skew"] += request_cv_excess * 1.0

    adjusted = {
        component: base_weights.raw_dict()[component] * factors[component]
        for component in SchedulerWeights.components()
    }
    tuned_weights = SchedulerWeights.from_dict(adjusted)
    signals = {
        "avg_active_ratio": round(avg_active_ratio, 4),
        "max_active_ratio": round(max_active_ratio, 4),
        "avg_gpu_pressure": round(avg_gpu_pressure, 4),
        "max_gpu_pressure": round(max_gpu_pressure, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "max_latency_ms": round(max_latency_ms, 2),
        "avg_ttft_ms": round(avg_ttft_ms, 2),
        "max_ttft_ms": round(max_ttft_ms, 2),
        "avg_throughput_tps": round(avg_throughput, 2),
        "min_throughput_tps": round(min_throughput, 2),
        "throughput_cv": round(throughput_cv, 4),
        "avg_failure_rate": round(avg_failure_rate, 4),
        "max_failure_rate": round(max_failure_rate, 4),
        "max_consecutive_failures": float(max_consecutive_failures),
        "request_cv": round(request_cv, 4),
    }
    return SchedulerTuningResult(weights=tuned_weights, signals=signals)


def _metric_penalty(value: float, baseline: float) -> float:
    if value <= 0 or baseline <= 0:
        return _NEUTRAL_METRIC_PENALTY
    return min(value / baseline, 2.5)


def _throughput_penalty(value: float, baseline: float) -> float:
    if value <= 0 or baseline <= 0:
        return _NEUTRAL_METRIC_PENALTY
    return min(baseline / value, 2.5)


def compute_candidate_score(
    *,
    active_ratio: float,
    gpu_pressure: float,
    latency_ms: float,
    latency_baseline_ms: float,
    ttft_ms: float,
    ttft_baseline_ms: float,
    throughput_tokens_per_second: float,
    throughput_baseline_tokens_per_second: float,
    recent_requests: int,
    total_recent_requests: int,
    healthy_candidates: int,
    failure_rate: float,
    consecutive_failures: int,
    cooldown_remaining_sec: float,
    weights: SchedulerWeights,
) -> tuple[float, dict[str, float]]:
    healthy_candidates = max(healthy_candidates, 1)
    recent_share = recent_requests / max(total_recent_requests, 1)
    fair_share = 1.0 / healthy_candidates
    skew_penalty = max(0.0, recent_share - fair_share) * healthy_candidates
    failure_penalty = min(
        2.5,
        max(failure_rate, 0.0) * 1.2
        + consecutive_failures * 0.25
        + min(
            cooldown_remaining_sec / max(DEFAULT_FAILURE_COOLDOWN_SEC, 1e-6),
            1.0,
        ),
    )
    components = {
        "load": min(max(active_ratio, 0.0), 1.0),
        "gpu_pressure": min(max(gpu_pressure, 0.0), 1.0),
        "latency": _metric_penalty(latency_ms, latency_baseline_ms),
        "ttft": _metric_penalty(ttft_ms, ttft_baseline_ms),
        "throughput": _throughput_penalty(
            throughput_tokens_per_second,
            throughput_baseline_tokens_per_second,
        ),
        "failures": failure_penalty,
        "skew": min(skew_penalty, 2.0),
    }
    score = (
        weights.load * components["load"]
        + weights.gpu_pressure * components["gpu_pressure"]
        + weights.latency * components["latency"]
        + weights.ttft * components["ttft"]
        + weights.throughput * components["throughput"]
        + weights.failures * components["failures"]
        + weights.skew * components["skew"]
    )
    return score, components
