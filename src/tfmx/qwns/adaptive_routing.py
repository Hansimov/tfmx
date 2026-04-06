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
