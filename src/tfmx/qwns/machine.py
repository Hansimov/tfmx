"""QWN machine proxy server and daemon management."""

import argparse
import asyncio
import os
import re
import signal
import subprocess
import sys
import time

import httpx
import orjson
import uvicorn

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pathlib import Path
from pydantic import BaseModel, Field
from tclogger import logger, logstr
from typing import Literal, Optional, Union
from webu import setup_swagger_ui

from .compose import MACHINE_PORT, MAX_CONCURRENT_REQUESTS, MAX_MODEL_LEN
from .compose import get_display_shortcut, get_model_shortcut, normalize_model_key
from .gpu_runtime import GPURuntimeStats, query_gpu_runtime_stats
from .router import InstanceDescriptor, QWNRouter, parse_model_spec
from .adaptive_routing import DEFAULT_FAILURE_COOLDOWN_SEC
from .adaptive_routing import InstanceTelemetry
from .adaptive_routing import RECENT_WINDOW_SEC
from .adaptive_routing import SchedulerTuningConfig
from .adaptive_routing import SchedulerTuningResult
from .adaptive_routing import SchedulerWeights
from .adaptive_routing import compute_candidate_score
from .adaptive_routing import get_peer_baseline
from .adaptive_routing import tune_scheduler_weights


PORT = MACHINE_PORT
MAX_CONCURRENT = MAX_CONCURRENT_REQUESTS
DEFAULT_MAX_TOKENS = MAX_MODEL_LEN
QWN_CONTAINER_IMAGE_PATTERN = "vllm"
DEFAULT_NAME_PATTERN = r"qwn[-_]"
HEALTH_REFRESH_INTERVAL_SEC = 5
HEALTH_REQUEST_TIMEOUT_SEC = 5.0
MODEL_DISCOVERY_TIMEOUT_SEC = 10.0
INSTANCE_ACQUIRE_TIMEOUT_SEC = 5.0
SCHEDULER_ALGORITHM = "adaptive_pressure_v2"


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")


class ImageURL(BaseModel):
    url: str = Field(
        ...,
        description="Image URL or base64 data URI (data:image/png;base64,...)",
    )
    detail: str = Field(default="auto", description="Detail level: auto, low, high")


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ContentPart = Union[TextContent, ImageContent]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Message role"
    )
    content: Union[str, list[ContentPart]] = Field(
        ...,
        description=(
            "Message content: plain text string, or a list of text/image_url parts "
            "for multimodal inputs"
        ),
    )


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str = Field(default="", description="Model label or shortcut")
    messages: list[ChatMessage] = Field(..., description="OpenAI chat messages")
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Enable streaming")


class ChatChoiceDelta(BaseModel):
    index: int = 0
    message: dict = Field(default_factory=dict)
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default="")
    object: str = Field(default="chat.completion")
    created: int = Field(default=0)
    model: str = Field(default="")
    choices: list[ChatChoiceDelta] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class InstanceHealthDetail(BaseModel):
    name: str = Field(..., description="Container or instance name")
    endpoint: str = Field(..., description="HTTP endpoint URL")
    healthy: bool = Field(..., description="Whether instance is healthy")
    latency_ms: float | None = Field(None, description="Last health check latency")
    active_requests: int = Field(0, description="Current active requests")
    model_label: str = Field(default="", description="Model label")
    gpu_id: int | None = Field(None, description="GPU ID")
    gpu_utilization_pct: float | None = Field(None, description="GPU utilization")
    gpu_memory_used_mib: float | None = Field(None, description="GPU memory used")
    gpu_memory_total_mib: float | None = Field(None, description="GPU memory total")
    routing_pressure: float | None = Field(
        None,
        description="Scheduler pressure score (lower is better)",
    )


class HealthResponse(BaseModel):
    status: str = Field(...)
    healthy: int = Field(...)
    total: int = Field(...)
    instances: list[InstanceHealthDetail] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str = Field(...)
    object: str = Field(default="model")
    created: int = Field(default=0)
    owned_by: str = Field(default="qwn-machine")


class ModelsResponse(BaseModel):
    object: str = Field(default="list")
    data: list[ModelInfo] = Field(default_factory=list)


class InstanceSchedulerInfo(BaseModel):
    score: float | None = Field(None)
    recent_requests: int = Field(0)
    recent_successes: int = Field(0)
    recent_failures: int = Field(0)
    success_rate: float | None = Field(None)
    latency_ema_ms: float | None = Field(None)
    ttft_ema_ms: float | None = Field(None)
    tokens_per_second_ema: float | None = Field(None)
    cooldown_remaining_sec: float = Field(0.0)
    consecutive_failures: int = Field(0)
    score_components: dict[str, float] = Field(default_factory=dict)
    last_error: str = Field(default="")


class InstanceInfo(BaseModel):
    name: str = Field(...)
    endpoint: str = Field(...)
    gpu_id: int | None = Field(None)
    healthy: bool = Field(...)
    model_name: str = Field(default="")
    quant_method: str = Field(default="")
    quant_level: str = Field(default="")
    model_label: str = Field(default="")
    active_requests: int = Field(0)
    available_slots: int = Field(0)
    gpu_utilization_pct: float | None = Field(None)
    gpu_memory_used_mib: float | None = Field(None)
    gpu_memory_total_mib: float | None = Field(None)
    routing_pressure: float | None = Field(None)
    scheduler: InstanceSchedulerInfo = Field(default_factory=InstanceSchedulerInfo)


class MachineStats(BaseModel):
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    total_failovers: int = 0
    active_requests: int = 0
    requests_per_instance: dict[str, int] = Field(default_factory=dict)
    total_wait_events: int = 0
    avg_wait_time_ms: float | None = Field(None)
    max_wait_time_ms: float | None = Field(None)


class SchedulerInfo(BaseModel):
    algorithm: str = Field(default=SCHEDULER_ALGORITHM)
    recent_window_sec: float = Field(default=RECENT_WINDOW_SEC)
    acquire_timeout_sec: float = Field(default=INSTANCE_ACQUIRE_TIMEOUT_SEC)
    last_health_refresh_age_sec: float | None = Field(None)
    last_gpu_refresh_age_sec: float | None = Field(None)
    weights: dict[str, float] = Field(default_factory=dict)
    base_weights: dict[str, float] = Field(default_factory=dict)
    tuning: dict[str, object] = Field(default_factory=dict)


class InfoResponse(BaseModel):
    port: int
    instances: list[InstanceInfo] = Field(default_factory=list)
    stats: MachineStats = Field(default_factory=MachineStats)
    available_models: list[str] = Field(default_factory=list)
    scheduler: SchedulerInfo = Field(default_factory=SchedulerInfo)


@dataclass
class QWNInstance:
    container_name: str
    host: str
    port: int
    gpu_id: Optional[int] = None
    healthy: bool = False
    _active_requests: int = 0
    model_name: str = ""
    quant_method: str = ""
    quant_level: str = ""
    _latency_ms: float = 0.0
    _gpu_utilization_pct: float = 0.0
    _gpu_memory_used_mib: float = 0.0
    _gpu_memory_total_mib: float = 0.0
    telemetry: InstanceTelemetry = field(default_factory=InstanceTelemetry)

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def chat_url(self) -> str:
        return f"{self.endpoint}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        return f"{self.endpoint}/health"

    @property
    def models_url(self) -> str:
        return f"{self.endpoint}/v1/models"

    @property
    def is_idle(self) -> bool:
        return self._active_requests < MAX_CONCURRENT

    @property
    def available_slots(self) -> int:
        return max(0, MAX_CONCURRENT - self._active_requests)

    def __repr__(self) -> str:
        status = "✓" if self.healthy else "×"
        gpu_text = f"GPU{self.gpu_id}" if self.gpu_id is not None else "GPU?"
        model_text = f" [{self.model_name}]" if self.model_name else ""
        return f"QWNInstance({status} {self.container_name} @ {self.endpoint}, {gpu_text}{model_text})"

    @property
    def gpu_memory_utilization_pct(self) -> float:
        if self._gpu_memory_total_mib <= 0:
            return 0.0
        return self._gpu_memory_used_mib / self._gpu_memory_total_mib * 100.0

    @property
    def routing_pressure(self) -> float:
        util_ratio = min(max(self._gpu_utilization_pct, 0.0), 100.0) / 100.0
        memory_ratio = min(max(self.gpu_memory_utilization_pct, 0.0), 100.0) / 100.0
        return util_ratio * 0.8 + memory_ratio * 0.2

    def apply_gpu_runtime(self, snapshot: GPURuntimeStats | None) -> None:
        if snapshot is None:
            self._gpu_utilization_pct = 0.0
            self._gpu_memory_used_mib = 0.0
            self._gpu_memory_total_mib = 0.0
            return
        self._gpu_utilization_pct = snapshot.utilization_gpu_pct
        self._gpu_memory_used_mib = snapshot.memory_used_mib
        self._gpu_memory_total_mib = snapshot.memory_total_mib

    def to_info(
        self,
        model_label: str = "",
        scheduler: InstanceSchedulerInfo | None = None,
    ) -> InstanceInfo:
        return InstanceInfo(
            name=self.container_name,
            endpoint=self.endpoint,
            gpu_id=self.gpu_id,
            healthy=self.healthy,
            model_name=self.model_name,
            quant_method=self.quant_method,
            quant_level=self.quant_level,
            model_label=model_label or self.model_name,
            active_requests=self._active_requests,
            available_slots=self.available_slots,
            gpu_utilization_pct=round(self._gpu_utilization_pct, 2),
            gpu_memory_used_mib=round(self._gpu_memory_used_mib, 2),
            gpu_memory_total_mib=round(self._gpu_memory_total_mib, 2),
            routing_pressure=round(self.routing_pressure, 4),
            scheduler=scheduler or InstanceSchedulerInfo(),
        )


@dataclass
class QWNStatsData:
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    total_failovers: int = 0
    active_requests: int = 0
    requests_per_instance: dict[str, int] = field(default_factory=dict)
    total_wait_events: int = 0
    total_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0

    def record_wait(self, wait_ms: float) -> None:
        wait_ms = max(wait_ms, 0.0)
        if wait_ms <= 0:
            return
        self.total_wait_events += 1
        self.total_wait_time_ms += wait_ms
        self.max_wait_time_ms = max(self.max_wait_time_ms, wait_ms)

    def to_model(self) -> MachineStats:
        avg_wait_time_ms = None
        if self.total_wait_events > 0:
            avg_wait_time_ms = self.total_wait_time_ms / self.total_wait_events
        return MachineStats(
            total_requests=self.total_requests,
            total_tokens=self.total_tokens,
            total_errors=self.total_errors,
            total_failovers=self.total_failovers,
            active_requests=self.active_requests,
            requests_per_instance=dict(self.requests_per_instance),
            total_wait_events=self.total_wait_events,
            avg_wait_time_ms=(
                round(avg_wait_time_ms, 2) if avg_wait_time_ms is not None else None
            ),
            max_wait_time_ms=(
                round(self.max_wait_time_ms, 2) if self.max_wait_time_ms > 0 else None
            ),
        )


class QWNInstanceDiscovery:
    @staticmethod
    def discover(name_pattern: Optional[str] = None) -> list[QWNInstance]:
        try:
            result = subprocess.run(
                "docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}'",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warn(f"× Docker command failed: {result.stderr}")
                return []
            if not result.stdout.strip():
                return []

            pattern = name_pattern or DEFAULT_NAME_PATTERN
            instances: list[QWNInstance] = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                container_name, image, ports = parts[0], parts[1], parts[2]
                if QWN_CONTAINER_IMAGE_PATTERN not in image:
                    continue
                if not re.search(pattern, container_name):
                    continue
                host_port = QWNInstanceDiscovery._extract_host_port(
                    ports, container_name
                )
                if host_port is None:
                    continue
                gpu_id = QWNInstanceDiscovery._extract_gpu_id(container_name)
                instances.append(
                    QWNInstance(
                        container_name=container_name,
                        host="localhost",
                        port=host_port,
                        gpu_id=gpu_id,
                    )
                )
            instances.sort(
                key=lambda instance: (
                    instance.gpu_id if instance.gpu_id is not None else 999
                )
            )
            return instances
        except Exception as exc:
            logger.warn(f"× Failed to discover QWN instances: {exc}")
            return []

    @staticmethod
    def _extract_host_port(ports_str: str, container_name: str = "") -> Optional[int]:
        match = re.search(r"(?:0\.0\.0\.0|::):(\d+)->", ports_str)
        if match:
            return int(match.group(1))

        if not ports_str and container_name:
            try:
                result = subprocess.run(
                    ["docker", "inspect", container_name, "--format", "{{.Args}}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    match = re.search(r"--port\s+(\d+)", result.stdout.strip())
                    if match:
                        return int(match.group(1))
            except Exception:
                pass
        return None

    @staticmethod
    def _extract_gpu_id(container_name: str) -> Optional[int]:
        match = re.search(r"--gpu(\d+)", container_name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def from_endpoints(endpoints: list[str]) -> list[QWNInstance]:
        instances: list[QWNInstance] = []
        for index, endpoint in enumerate(endpoints):
            endpoint = endpoint.strip()
            if not endpoint:
                continue
            match = re.match(r"https?://([^:]+):(\d+)", endpoint)
            if match:
                host, port = match.group(1), int(match.group(2))
            else:
                try:
                    host, port = "localhost", int(endpoint)
                except ValueError:
                    continue
            instances.append(
                QWNInstance(
                    container_name=f"manual-{index}",
                    host=host,
                    port=port,
                    gpu_id=index,
                )
            )
        return instances


class QWNMachineServer:
    def __init__(
        self,
        instances: list[QWNInstance],
        port: int = PORT,
        timeout: float = 120.0,
    ):
        self.instances = instances
        self.port = port
        self.timeout = timeout
        self.stats = QWNStatsData()
        self._client: Optional[httpx.AsyncClient] = None
        self._health_task: Optional[asyncio.Task] = None
        self.router = QWNRouter()
        self.scheduler_base_weights = SchedulerWeights()
        self.scheduler_tuning_config = SchedulerTuningConfig()
        self._capacity_cond: Optional[asyncio.Condition] = None
        self._last_health_refresh_monotonic: float = 0.0
        self._last_gpu_refresh_monotonic: float = 0.0
        self.app = self._create_app()

    def _build_router(self) -> None:
        self.router = QWNRouter()
        for instance in self.instances:
            self.router.register(
                InstanceDescriptor(
                    model_name=instance.model_name,
                    quant_method=instance.quant_method,
                    quant_level=instance.quant_level,
                    endpoint=instance.endpoint,
                    gpu_id=instance.gpu_id,
                    instance_id=instance.container_name,
                    healthy=instance.healthy,
                )
            )

    def get_healthy_instances(self) -> list[QWNInstance]:
        return [instance for instance in self.instances if instance.healthy]

    def _ensure_scheduler_primitives(self) -> None:
        if self._capacity_cond is None:
            self._capacity_cond = asyncio.Condition()

    async def _notify_capacity_changed(self) -> None:
        self._ensure_scheduler_primitives()
        async with self._capacity_cond:
            self._capacity_cond.notify_all()

    def _notify_capacity_changed_soon(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._notify_capacity_changed())

    def _get_candidate_instances(
        self,
        model: str = "",
        quant: str = "",
        excluded_instances: set[str] | None = None,
        require_idle: bool = False,
    ) -> list[QWNInstance]:
        excluded_instances = excluded_instances or set()
        if model or quant:
            descriptors = self.router.find_instances(model, quant)
            endpoints = {descriptor.endpoint for descriptor in descriptors}
            candidates = [
                instance
                for instance in self.instances
                if instance.healthy
                and instance.endpoint in endpoints
                and instance.container_name not in excluded_instances
            ]
        else:
            candidates = [
                instance
                for instance in self.instances
                if instance.healthy
                and instance.container_name not in excluded_instances
            ]
        if require_idle:
            candidates = [instance for instance in candidates if instance.is_idle]
        return candidates

    def _get_peer_instances_for_instance(
        self,
        instance: QWNInstance,
        healthy_instances: list[QWNInstance],
    ) -> list[QWNInstance]:
        if not healthy_instances:
            return []
        if not instance.model_name:
            return healthy_instances
        peer_instances = [
            peer
            for peer in healthy_instances
            if peer.model_name == instance.model_name
            and peer.quant_level == instance.quant_level
        ]
        return peer_instances or healthy_instances

    def _get_effective_scheduler_weights(
        self,
        peer_instances: list[QWNInstance],
        now: float,
    ) -> SchedulerTuningResult:
        if not peer_instances:
            return SchedulerTuningResult(
                weights=self.scheduler_base_weights,
                signals={},
            )

        active_ratios = [
            instance._active_requests / max(MAX_CONCURRENT, 1)
            for instance in peer_instances
        ]
        gpu_pressures = [instance.routing_pressure for instance in peer_instances]
        latencies_ms = [
            instance.telemetry.latency_ema_ms for instance in peer_instances
        ]
        ttfts_ms = [instance.telemetry.ttft_ema_ms for instance in peer_instances]
        throughputs_tokens_per_second = [
            instance.telemetry.tokens_per_second_ema for instance in peer_instances
        ]
        recent_requests = [
            instance.telemetry.recent_requests(now) for instance in peer_instances
        ]
        failure_rates = []
        consecutive_failures = []
        for instance in peer_instances:
            recent_request_count = instance.telemetry.recent_requests(now)
            recent_failure_count = instance.telemetry.recent_failures(now)
            failure_rates.append(
                recent_failure_count / recent_request_count
                if recent_request_count > 0
                else 0.0
            )
            consecutive_failures.append(instance.telemetry.consecutive_failures)

        return tune_scheduler_weights(
            base_weights=self.scheduler_base_weights,
            active_ratios=active_ratios,
            gpu_pressures=gpu_pressures,
            latencies_ms=latencies_ms,
            ttfts_ms=ttfts_ms,
            throughputs_tokens_per_second=throughputs_tokens_per_second,
            recent_requests=recent_requests,
            failure_rates=failure_rates,
            consecutive_failures=consecutive_failures,
            config=self.scheduler_tuning_config,
        )

    def _score_instance(
        self,
        instance: QWNInstance,
        peer_instances: list[QWNInstance],
        now: float,
        weights: SchedulerWeights,
    ) -> tuple[float, dict[str, float]]:
        telemetry = instance.telemetry.snapshot(now)
        recent_requests = telemetry["recent_requests"]
        recent_failures = telemetry["recent_failures"]
        failure_rate = recent_failures / recent_requests if recent_requests > 0 else 0.0
        latency_values = [
            peer.telemetry.latency_ema_ms
            for peer in peer_instances
            if peer.telemetry.latency_ema_ms > 0
        ]
        ttft_values = [
            peer.telemetry.ttft_ema_ms
            for peer in peer_instances
            if peer.telemetry.ttft_ema_ms > 0
        ]
        throughput_values = [
            peer.telemetry.tokens_per_second_ema
            for peer in peer_instances
            if peer.telemetry.tokens_per_second_ema > 0
        ]
        total_recent_requests = sum(
            peer.telemetry.recent_requests(now) for peer in peer_instances
        )
        return compute_candidate_score(
            active_ratio=instance._active_requests / max(MAX_CONCURRENT, 1),
            gpu_pressure=instance.routing_pressure,
            latency_ms=instance.telemetry.latency_ema_ms,
            latency_baseline_ms=get_peer_baseline(latency_values, 2500.0),
            ttft_ms=instance.telemetry.ttft_ema_ms,
            ttft_baseline_ms=get_peer_baseline(ttft_values, 1500.0),
            throughput_tokens_per_second=instance.telemetry.tokens_per_second_ema,
            throughput_baseline_tokens_per_second=get_peer_baseline(
                throughput_values,
                40.0,
            ),
            recent_requests=recent_requests,
            total_recent_requests=total_recent_requests,
            healthy_candidates=len(peer_instances),
            failure_rate=failure_rate,
            consecutive_failures=instance.telemetry.consecutive_failures,
            cooldown_remaining_sec=instance.telemetry.cooldown_remaining_sec(now),
            weights=weights,
        )

    def _build_instance_scheduler_info(
        self,
        instance: QWNInstance,
        peer_instances: list[QWNInstance],
        now: float,
        weights: SchedulerWeights,
    ) -> InstanceSchedulerInfo:
        telemetry = instance.telemetry.snapshot(now)
        score = None
        components: dict[str, float] = {}
        if instance.healthy and peer_instances:
            score, components = self._score_instance(
                instance,
                peer_instances,
                now,
                weights,
            )
        return InstanceSchedulerInfo(
            score=round(score, 4) if score is not None else None,
            recent_requests=telemetry["recent_requests"],
            recent_successes=telemetry["recent_successes"],
            recent_failures=telemetry["recent_failures"],
            success_rate=telemetry["success_rate"],
            latency_ema_ms=telemetry["latency_ema_ms"],
            ttft_ema_ms=telemetry["ttft_ema_ms"],
            tokens_per_second_ema=telemetry["tokens_per_second_ema"],
            cooldown_remaining_sec=telemetry["cooldown_remaining_sec"],
            consecutive_failures=telemetry["consecutive_failures"],
            score_components={
                key: round(value, 4) for key, value in components.items()
            },
            last_error=telemetry["last_error"],
        )

    def _build_scheduler_summary(self) -> SchedulerInfo:
        now = time.monotonic()
        healthy_instances = self.get_healthy_instances()
        tuning = self._get_effective_scheduler_weights(healthy_instances, now)
        return SchedulerInfo(
            algorithm=SCHEDULER_ALGORITHM,
            recent_window_sec=RECENT_WINDOW_SEC,
            acquire_timeout_sec=INSTANCE_ACQUIRE_TIMEOUT_SEC,
            last_health_refresh_age_sec=(
                round(
                    now - self._last_health_refresh_monotonic,
                    2,
                )
                if self._last_health_refresh_monotonic > 0
                else None
            ),
            last_gpu_refresh_age_sec=(
                round(
                    now - self._last_gpu_refresh_monotonic,
                    2,
                )
                if self._last_gpu_refresh_monotonic > 0
                else None
            ),
            weights=tuning.weights.to_dict(),
            base_weights=self.scheduler_base_weights.to_dict(),
            tuning={
                **self.scheduler_tuning_config.to_dict(),
                "signals": tuning.signals,
            },
        )

    def _select_idle_instance(
        self,
        model: str = "",
        quant: str = "",
        excluded_instances: set[str] | None = None,
    ) -> tuple[QWNInstance | None, bool]:
        candidates = self._get_candidate_instances(
            model=model,
            quant=quant,
            excluded_instances=excluded_instances,
            require_idle=False,
        )
        if not candidates:
            return None, False
        idle_candidates = [instance for instance in candidates if instance.is_idle]
        if not idle_candidates:
            return None, True
        now = time.monotonic()
        tuning = self._get_effective_scheduler_weights(candidates, now)
        scored_candidates: list[tuple[float, QWNInstance]] = []
        for instance in idle_candidates:
            score, _ = self._score_instance(
                instance,
                candidates,
                now,
                tuning.weights,
            )
            scored_candidates.append((score, instance))
        scored_candidates.sort(
            key=lambda item: (
                item[0],
                -item[1].available_slots,
                item[1].routing_pressure,
                item[1].gpu_id if item[1].gpu_id is not None else 999,
            )
        )
        return scored_candidates[0][1], True

    def _reserve_idle_instance_locked(
        self,
        model: str = "",
        quant: str = "",
        excluded_instances: set[str] | None = None,
    ) -> tuple[QWNInstance | None, bool]:
        instance, has_candidates = self._select_idle_instance(
            model=model,
            quant=quant,
            excluded_instances=excluded_instances,
        )
        if instance is None:
            return None, has_candidates
        instance._active_requests += 1
        self.stats.requests_per_instance[instance.container_name] = (
            self.stats.requests_per_instance.get(instance.container_name, 0) + 1
        )
        instance.telemetry.record_dispatch()
        return instance, True

    def _get_idle_instance(
        self,
        model: str = "",
        quant: str = "",
        excluded_instances: set[str] | None = None,
    ) -> Optional[QWNInstance]:
        instance, _ = self._select_idle_instance(
            model=model,
            quant=quant,
            excluded_instances=excluded_instances,
        )
        return instance

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="QWN Machine",
            description=(
                "Load-balanced proxy for Qwen 3.5 text vLLM instances.\n\n"
                "- POST /v1/chat/completions: OpenAI-compatible chat API\n"
                "- GET /v1/models: list available models\n"
                "- POST /chat: simplified form chat endpoint\n"
                "- GET /health: instance health summary\n"
                "- GET /info: machine stats and instance metadata"
            ),
            version="1.0.0",
            lifespan=self._lifespan,
            docs_url=None,
            redoc_url=None,
        )
        setup_swagger_ui(app)

        @app.exception_handler(HTTPException)
        async def openai_exception_handler(request: Request, exc: HTTPException):
            detail = exc.detail
            if isinstance(detail, dict):
                return JSONResponse(status_code=exc.status_code, content=detail)
            error_types = {
                400: "invalid_request_error",
                401: "authentication_error",
                403: "permission_error",
                404: "not_found_error",
                422: "invalid_request_error",
                429: "rate_limit_error",
                500: "server_error",
                503: "service_unavailable",
            }
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": str(detail),
                        "type": error_types.get(exc.status_code, "server_error"),
                        "code": exc.status_code,
                    }
                },
            )

        app.get("/health", response_model=HealthResponse)(self.health)
        app.get("/v1/models", response_model=ModelsResponse)(self.models)
        app.get("/info", response_model=InfoResponse)(self.info)
        app.get("/metrics")(self.metrics)
        app.post("/v1/chat/completions", response_model=ChatCompletionResponse)(
            self.chat_completions
        )
        app.post("/chat", response_model=ChatCompletionResponse)(self.chat_form)
        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        self._ensure_scheduler_primitives()

        await self.health_check_all()
        await self._discover_instance_models()
        self._build_router()

        healthy = self.get_healthy_instances()
        logger.okay(f"[qwn_machine] Started on port {self.port}")
        logger.mesg(
            f"[qwn_machine] Healthy instances: {logstr.okay(len(healthy))}/{logstr.mesg(len(self.instances))}"
        )
        available_models = self.router.get_available_models()
        if available_models:
            logger.mesg(f"[qwn_machine] Models: {available_models}")

        self._health_task = asyncio.create_task(self._periodic_health_check())
        yield

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()

    async def _periodic_health_check(self) -> None:
        while True:
            await asyncio.sleep(HEALTH_REFRESH_INTERVAL_SEC)
            await self.health_check_all()

    async def health_check_all(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        await asyncio.gather(
            *[self._check_instance_health(instance) for instance in self.instances],
            return_exceptions=True,
        )
        await self._refresh_gpu_runtime_metrics()
        await self._discover_instance_models()
        self._build_router()
        self._last_health_refresh_monotonic = time.monotonic()
        await self._notify_capacity_changed()

    async def _refresh_gpu_runtime_metrics(self) -> None:
        gpu_ids = [
            instance.gpu_id
            for instance in self.instances
            if instance.gpu_id is not None
        ]
        if not gpu_ids:
            return
        snapshots = await asyncio.to_thread(query_gpu_runtime_stats, gpu_ids)
        for instance in self.instances:
            if instance.gpu_id is None:
                continue
            instance.apply_gpu_runtime(snapshots.get(instance.gpu_id))
        self._last_gpu_refresh_monotonic = time.monotonic()

    async def _check_instance_health(self, instance: QWNInstance) -> bool:
        was_healthy = instance.healthy
        try:
            started_at = time.perf_counter()
            response = await self._client.get(
                instance.health_url,
                timeout=httpx.Timeout(HEALTH_REQUEST_TIMEOUT_SEC),
            )
            instance.healthy = response.status_code == 200
            instance._latency_ms = (
                (time.perf_counter() - started_at) * 1000 if instance.healthy else 0.0
            )
            if instance.healthy and not was_healthy:
                instance.telemetry.record_recovery()
                await self._notify_capacity_changed()
            elif not instance.healthy and was_healthy:
                instance.telemetry.record_failure("health check failed")
                await self._notify_capacity_changed()
            return instance.healthy
        except Exception:
            instance.healthy = False
            instance._latency_ms = 0.0
            if was_healthy:
                instance.telemetry.record_failure("health check failed")
                await self._notify_capacity_changed()
            return False

    async def _discover_instance_model(self, instance: QWNInstance) -> None:
        try:
            response = await self._client.get(
                instance.models_url,
                timeout=httpx.Timeout(MODEL_DISCOVERY_TIMEOUT_SEC),
            )
            if response.status_code != 200:
                return
            payload = response.json()
            models = payload.get("data", [])
            if not models:
                return
            model_id = models[0].get("id", "")
            instance.model_name = model_id
            base_model, quant_level = parse_model_spec(model_id)
            if quant_level:
                instance.quant_method = "awq"
                instance.quant_level = quant_level
            elif "awq" in model_id.lower():
                instance.quant_method = "awq"
            logger.mesg(f"[qwn_machine] {instance.container_name}: model={model_id}")
        except Exception:
            pass

    async def _discover_instance_models(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        pending = [
            self._discover_instance_model(instance)
            for instance in self.instances
            if instance.healthy and not instance.model_name
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _get_model_label(self, instance: QWNInstance) -> str:
        if instance.model_name and ":" in instance.model_name:
            return instance.model_name
        shortcut = get_model_shortcut(instance.model_name)
        display = get_display_shortcut(shortcut) if shortcut else ""
        if display and instance.quant_level:
            return f"{display}:{instance.quant_level}"
        return display or instance.model_name

    async def health(self) -> HealthResponse:
        healthy_instances = self.get_healthy_instances()
        payload = HealthResponse(
            status="healthy" if healthy_instances else "unhealthy",
            healthy=len(healthy_instances),
            total=len(self.instances),
            instances=[
                InstanceHealthDetail(
                    name=instance.container_name,
                    endpoint=instance.endpoint,
                    healthy=instance.healthy,
                    latency_ms=(
                        round(instance._latency_ms, 2)
                        if instance._latency_ms > 0
                        else None
                    ),
                    active_requests=instance._active_requests,
                    model_label=self._get_model_label(instance),
                    gpu_id=instance.gpu_id,
                    gpu_utilization_pct=round(instance._gpu_utilization_pct, 2),
                    gpu_memory_used_mib=round(instance._gpu_memory_used_mib, 2),
                    gpu_memory_total_mib=round(instance._gpu_memory_total_mib, 2),
                    routing_pressure=round(instance.routing_pressure, 4),
                )
                for instance in self.instances
            ],
        )
        if not healthy_instances:
            raise HTTPException(status_code=503, detail=payload.model_dump())
        return payload

    async def models(self) -> ModelsResponse:
        seen: set[str] = set()
        data: list[ModelInfo] = []
        created = int(time.time())
        for instance in self.instances:
            if not instance.healthy:
                continue
            label = self._get_model_label(instance)
            if label and label not in seen:
                seen.add(label)
                data.append(
                    ModelInfo(id=label, created=created, owned_by="qwn-machine")
                )
        return ModelsResponse(data=data)

    async def info(self) -> InfoResponse:
        now = time.monotonic()
        healthy_instances = self.get_healthy_instances()
        return InfoResponse(
            port=self.port,
            instances=[
                instance.to_info(
                    model_label=self._get_model_label(instance),
                    scheduler=self._build_instance_scheduler_info(
                        instance,
                        self._get_peer_instances_for_instance(
                            instance,
                            healthy_instances,
                        ),
                        now,
                        self._get_effective_scheduler_weights(
                            self._get_peer_instances_for_instance(
                                instance,
                                healthy_instances,
                            ),
                            now,
                        ).weights,
                    ),
                )
                for instance in self.instances
            ],
            stats=self.stats.to_model(),
            available_models=self.router.get_available_models(),
            scheduler=self._build_scheduler_summary(),
        )

    async def chat_completions(self, request: ChatCompletionRequest):
        body = orjson.dumps(request.model_dump(exclude_none=True))
        if request.stream:
            return await self._forward_stream(body, request.model)
        return await self._forward_chat(body, request.model)

    async def chat_form(
        self,
        text: str = Form(..., description="Prompt text"),
        system_prompt: str = Form(default="", description="Optional system prompt"),
        model: str = Form(default="", description="Model label"),
        max_tokens: int = Form(
            default=DEFAULT_MAX_TOKENS, description="Maximum tokens"
        ),
        temperature: float = Form(default=0.7, description="Temperature"),
        top_p: float = Form(default=0.9, description="Top-p"),
    ) -> ChatCompletionResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        body = orjson.dumps(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return await self._forward_chat(body, model)

    async def _acquire_instance(
        self,
        model_field: str = "",
        excluded_instances: set[str] | None = None,
    ) -> tuple[QWNInstance, str, str]:
        excluded_instances = excluded_instances or set()
        requested_model = ""
        requested_quant = ""
        if model_field:
            requested_model, requested_quant = parse_model_spec(model_field)
        self._ensure_scheduler_primitives()
        waited_sec = 0.0
        deadline = time.perf_counter() + INSTANCE_ACQUIRE_TIMEOUT_SEC

        async with self._capacity_cond:
            while True:
                instance, has_candidates = self._reserve_idle_instance_locked(
                    model=requested_model,
                    quant=requested_quant,
                    excluded_instances=excluded_instances,
                )
                if instance is not None:
                    if waited_sec > 0:
                        self.stats.record_wait(waited_sec * 1000.0)
                    return instance, requested_model, requested_quant

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    if waited_sec > 0:
                        self.stats.record_wait(waited_sec * 1000.0)
                    detail = "All QWN instances are busy"
                    if model_field:
                        detail = f"All QWN instances for requested model '{model_field}' are busy"
                    raise HTTPException(status_code=503, detail=detail)

                if not has_candidates:
                    if waited_sec > 0:
                        self.stats.record_wait(waited_sec * 1000.0)
                    detail = "No available QWN instances"
                    if model_field:
                        detail = f"No available QWN instances for requested model '{model_field}'"
                    raise HTTPException(status_code=503, detail=detail)

                wait_started_at = time.perf_counter()
                try:
                    await asyncio.wait_for(
                        self._capacity_cond.wait(),
                        timeout=min(remaining, 0.5),
                    )
                except asyncio.TimeoutError:
                    pass
                waited_sec += time.perf_counter() - wait_started_at

    def _rewrite_model_in_body(self, body: bytes, instance: QWNInstance) -> bytes:
        if not instance.model_name:
            return body
        try:
            payload = orjson.loads(body)
            payload["model"] = instance.model_name
            return orjson.dumps(payload)
        except Exception:
            return body

    def _is_retryable_upstream_status(self, status_code: int) -> bool:
        return status_code >= 500

    def _is_retryable_upstream_exception(self, exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                httpx.TransportError,
                httpx.TimeoutException,
                asyncio.TimeoutError,
            ),
        )

    def _mark_instance_unhealthy(
        self, instance: QWNInstance, reason: Exception | str
    ) -> None:
        reason_text = str(reason)
        if instance.healthy:
            logger.warn(
                f"× Marking {instance.container_name} unhealthy for failover: {reason_text}"
            )
        instance.telemetry.record_failure(
            reason_text,
            cooldown_sec=DEFAULT_FAILURE_COOLDOWN_SEC,
        )
        instance.healthy = False
        instance._latency_ms = 0.0
        self._build_router()
        self._notify_capacity_changed_soon()

    async def _release_instance(self, instance: QWNInstance) -> None:
        instance._active_requests = max(0, instance._active_requests - 1)
        await self._notify_capacity_changed()

    def _build_stream_error_chunk(
        self,
        message: str,
        code: int,
        error_type: str,
    ) -> str:
        payload = {
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        }
        return f"data: {orjson.dumps(payload).decode()}\n\n"

    @staticmethod
    def _escape_prometheus_label(value: object) -> str:
        text = str(value)
        return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    @staticmethod
    def _format_prometheus_value(value: object) -> str:
        if value is None:
            return "NaN"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        return f"{float(value):.10g}"

    def _append_prometheus_metric(
        self,
        lines: list[str],
        declared: set[str],
        name: str,
        value: object,
        *,
        labels: dict[str, object] | None = None,
        help_text: str | None = None,
        metric_type: str | None = None,
    ) -> None:
        if name not in declared:
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
            if metric_type:
                lines.append(f"# TYPE {name} {metric_type}")
            declared.add(name)
        label_text = ""
        if labels:
            label_text = (
                "{"
                + ",".join(
                    f'{key}="{self._escape_prometheus_label(label_value)}"'
                    for key, label_value in sorted(labels.items())
                )
                + "}"
            )
        lines.append(f"{name}{label_text} {self._format_prometheus_value(value)}")

    def _build_metrics_payload(self) -> str:
        now = time.monotonic()
        healthy_instances = self.get_healthy_instances()
        stats_model = self.stats.to_model()
        scheduler_summary = self._build_scheduler_summary()
        tuning = self._get_effective_scheduler_weights(healthy_instances, now)

        lines: list[str] = []
        declared: set[str] = set()
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_up",
            True,
            help_text="Whether the qwn machine process is serving requests.",
            metric_type="gauge",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_instances_total",
            len(self.instances),
            help_text="Total number of discovered qwn backend instances.",
            metric_type="gauge",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_instances_healthy_total",
            len(healthy_instances),
            help_text="Number of currently healthy qwn backend instances.",
            metric_type="gauge",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_requests_total",
            stats_model.total_requests,
            help_text="Total requests handled by qwn machine.",
            metric_type="counter",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_tokens_total",
            stats_model.total_tokens,
            help_text="Total tokens reported by upstream qwn backends.",
            metric_type="counter",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_errors_total",
            stats_model.total_errors,
            help_text="Total proxy-visible qwn machine errors.",
            metric_type="counter",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_failovers_total",
            stats_model.total_failovers,
            help_text="Total pre-response failovers performed by qwn machine.",
            metric_type="counter",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_active_requests",
            stats_model.active_requests,
            help_text="Number of requests currently active in qwn machine.",
            metric_type="gauge",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_wait_events_total",
            stats_model.total_wait_events,
            help_text="Total number of instance-capacity wait events observed by qwn machine.",
            metric_type="counter",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_wait_time_avg_ms",
            stats_model.avg_wait_time_ms,
            help_text="Average instance-capacity wait time in milliseconds.",
            metric_type="gauge",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_wait_time_max_ms",
            stats_model.max_wait_time_ms,
            help_text="Maximum observed instance-capacity wait time in milliseconds.",
            metric_type="gauge",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_scheduler_auto_tuning_enabled",
            self.scheduler_tuning_config.enabled,
            help_text="Whether scheduler auto-tuning is enabled.",
            metric_type="gauge",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_scheduler_refresh_age_seconds",
            scheduler_summary.last_health_refresh_age_sec,
            labels={"kind": "health"},
            help_text="Age of the latest scheduler refresh snapshots.",
            metric_type="gauge",
        )
        self._append_prometheus_metric(
            lines,
            declared,
            "qwn_machine_scheduler_refresh_age_seconds",
            scheduler_summary.last_gpu_refresh_age_sec,
            labels={"kind": "gpu"},
            metric_type="gauge",
        )
        for component, weight in scheduler_summary.base_weights.items():
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_scheduler_weight",
                weight,
                labels={"kind": "base", "component": component},
                help_text="Scheduler component weights.",
                metric_type="gauge",
            )
        for component, weight in scheduler_summary.weights.items():
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_scheduler_weight",
                weight,
                labels={"kind": "effective", "component": component},
                metric_type="gauge",
            )
        for signal_name, signal_value in tuning.signals.items():
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_scheduler_signal",
                signal_value,
                labels={"signal": signal_name},
                help_text="Recent scheduler auto-tuning signals.",
                metric_type="gauge",
            )
        for (
            setting_name,
            setting_value,
        ) in self.scheduler_tuning_config.to_dict().items():
            if setting_name == "enabled":
                continue
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_scheduler_config",
                setting_value,
                labels={"setting": setting_name},
                help_text="Scheduler auto-tuning configuration values.",
                metric_type="gauge",
            )

        for instance in self.instances:
            peer_instances = self._get_peer_instances_for_instance(
                instance,
                healthy_instances,
            )
            peer_tuning = self._get_effective_scheduler_weights(peer_instances, now)
            scheduler_info = self._build_instance_scheduler_info(
                instance,
                peer_instances,
                now,
                peer_tuning.weights,
            )
            labels = {
                "instance": instance.container_name,
                "endpoint": instance.endpoint,
                "gpu_id": instance.gpu_id if instance.gpu_id is not None else "unknown",
                "model": self._get_model_label(instance) or "unknown",
            }
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_up",
                instance.healthy,
                labels=labels,
                help_text="Whether a qwn backend instance is currently healthy.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_requests_total",
                self.stats.requests_per_instance.get(instance.container_name, 0),
                labels=labels,
                help_text="Total requests routed to a qwn backend instance.",
                metric_type="counter",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_active_requests",
                instance._active_requests,
                labels=labels,
                help_text="Current active requests for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_available_slots",
                instance.available_slots,
                labels=labels,
                help_text="Currently available request slots for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_gpu_utilization_pct",
                instance._gpu_utilization_pct,
                labels=labels,
                help_text="Current GPU utilization percentage for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_gpu_memory_used_mib",
                instance._gpu_memory_used_mib,
                labels=labels,
                help_text="Current GPU memory used in MiB for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_gpu_memory_total_mib",
                instance._gpu_memory_total_mib,
                labels=labels,
                help_text="Total GPU memory in MiB for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_routing_pressure",
                instance.routing_pressure,
                labels=labels,
                help_text="Current routing pressure for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_scheduler_score",
                scheduler_info.score,
                labels=labels,
                help_text="Current scheduler score for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_recent_requests",
                scheduler_info.recent_requests,
                labels=labels,
                help_text="Recent-window requests seen by a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_recent_successes",
                scheduler_info.recent_successes,
                labels=labels,
                help_text="Recent-window successful requests for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_recent_failures",
                scheduler_info.recent_failures,
                labels=labels,
                help_text="Recent-window failed requests for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_success_rate",
                scheduler_info.success_rate,
                labels=labels,
                help_text="Recent-window success rate for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_latency_ema_ms",
                scheduler_info.latency_ema_ms,
                labels=labels,
                help_text="Latency EMA in milliseconds for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_ttft_ema_ms",
                scheduler_info.ttft_ema_ms,
                labels=labels,
                help_text="TTFT EMA in milliseconds for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_tokens_per_second_ema",
                scheduler_info.tokens_per_second_ema,
                labels=labels,
                help_text="Generation tokens-per-second EMA for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_cooldown_remaining_seconds",
                scheduler_info.cooldown_remaining_sec,
                labels=labels,
                help_text="Remaining scheduler cooldown for a qwn backend instance.",
                metric_type="gauge",
            )
            self._append_prometheus_metric(
                lines,
                declared,
                "qwn_machine_instance_consecutive_failures",
                scheduler_info.consecutive_failures,
                labels=labels,
                help_text="Current consecutive failure count for a qwn backend instance.",
                metric_type="gauge",
            )
            for component, component_value in scheduler_info.score_components.items():
                self._append_prometheus_metric(
                    lines,
                    declared,
                    "qwn_machine_instance_scheduler_component",
                    component_value,
                    labels={**labels, "component": component},
                    help_text="Per-component scheduler penalties for a qwn backend instance.",
                    metric_type="gauge",
                )

        return "\n".join(lines) + "\n"

    async def metrics(self) -> PlainTextResponse:
        return PlainTextResponse(
            self._build_metrics_payload(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def _forward_stream(
        self, body: bytes, model_field: str = ""
    ) -> StreamingResponse:
        self.stats.total_requests += 1
        self.stats.active_requests += 1

        async def stream_generator():
            try:
                attempted_instances: set[str] = set()
                last_retryable_error = ""
                while True:
                    try:
                        instance, _, _ = await self._acquire_instance(
                            model_field,
                            excluded_instances=attempted_instances,
                        )
                    except HTTPException as exc:
                        self.stats.total_errors += 1
                        detail = str(exc.detail)
                        if last_retryable_error and exc.status_code == 503:
                            detail = (
                                "All candidate QWN instances failed before "
                                f"stream start: {last_retryable_error}"
                            )
                        yield self._build_stream_error_chunk(
                            message=detail,
                            code=exc.status_code,
                            error_type="service_unavailable",
                        )
                        yield "data: [DONE]\n\n"
                        return

                    attempted_instances.add(instance.container_name)
                    model_label = self._get_model_label(instance)
                    attempt_body = self._rewrite_model_in_body(body, instance)
                    started_stream = False
                    started_at = time.perf_counter()
                    first_chunk_latency_ms = 0.0
                    completion_tokens = 0

                    try:
                        async with self._client.stream(
                            "POST",
                            instance.chat_url,
                            content=attempt_body,
                            headers={"Content-Type": "application/json"},
                        ) as response:
                            if response.status_code != 200:
                                error_text = (await response.aread()).decode(
                                    "utf-8", errors="replace"
                                )
                                if self._is_retryable_upstream_status(
                                    response.status_code
                                ):
                                    last_retryable_error = (
                                        f"{instance.container_name} returned "
                                        f"{response.status_code}: {error_text}"
                                    )
                                    self.stats.total_failovers += 1
                                    self._mark_instance_unhealthy(
                                        instance, last_retryable_error
                                    )
                                    continue
                                self.stats.total_errors += 1
                                yield self._build_stream_error_chunk(
                                    message=error_text,
                                    code=response.status_code,
                                    error_type="upstream_error",
                                )
                                yield "data: [DONE]\n\n"
                                return

                            async for line in response.aiter_lines():
                                if not line.strip():
                                    continue
                                if line.startswith("data: "):
                                    payload = line[6:].strip()
                                    if payload == "[DONE]":
                                        if started_stream:
                                            instance.telemetry.record_success(
                                                latency_ms=(
                                                    time.perf_counter() - started_at
                                                )
                                                * 1000.0,
                                                ttft_ms=first_chunk_latency_ms,
                                                completion_tokens=completion_tokens,
                                            )
                                        yield "data: [DONE]\n\n"
                                        return
                                    if first_chunk_latency_ms <= 0:
                                        first_chunk_latency_ms = (
                                            time.perf_counter() - started_at
                                        ) * 1000.0
                                    try:
                                        chunk = orjson.loads(payload)
                                        if model_label:
                                            chunk["model"] = model_label
                                        usage = chunk.get("usage")
                                        if usage:
                                            self.stats.total_tokens += usage.get(
                                                "total_tokens", 0
                                            )
                                            completion_tokens = usage.get(
                                                "completion_tokens",
                                                completion_tokens,
                                            )
                                        started_stream = True
                                        yield (
                                            f"data: {orjson.dumps(chunk).decode()}\n\n"
                                        )
                                    except Exception:
                                        started_stream = True
                                        yield f"data: {payload}\n\n"
                                elif line.startswith(":"):
                                    yield f"{line}\n"

                            if not started_stream:
                                last_retryable_error = (
                                    f"{instance.container_name} closed stream before "
                                    "sending any data"
                                )
                                self.stats.total_failovers += 1
                                self._mark_instance_unhealthy(
                                    instance, last_retryable_error
                                )
                                continue
                            instance.telemetry.record_success(
                                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                                ttft_ms=first_chunk_latency_ms,
                                completion_tokens=completion_tokens,
                            )
                            return
                    except Exception as exc:
                        if (
                            not started_stream
                            and self._is_retryable_upstream_exception(exc)
                        ):
                            last_retryable_error = f"{instance.container_name}: {exc}"
                            self.stats.total_failovers += 1
                            self._mark_instance_unhealthy(instance, exc)
                            continue
                        self.stats.total_errors += 1
                        logger.warn(f"× Stream error: {exc}")
                        yield self._build_stream_error_chunk(
                            message=str(exc),
                            code=500,
                            error_type="proxy_error",
                        )
                        yield "data: [DONE]\n\n"
                        return
                    finally:
                        await self._release_instance(instance)
            except Exception as exc:
                self.stats.total_errors += 1
                logger.warn(f"× Stream error: {exc}")
                yield self._build_stream_error_chunk(
                    message=str(exc),
                    code=500,
                    error_type="proxy_error",
                )
                yield "data: [DONE]\n\n"
            finally:
                self.stats.active_requests = max(0, self.stats.active_requests - 1)

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _forward_chat(
        self, body: bytes, model_field: str = ""
    ) -> ChatCompletionResponse:
        self.stats.total_requests += 1
        self.stats.active_requests += 1
        try:
            attempted_instances: set[str] = set()
            last_retryable_error = ""
            while True:
                try:
                    instance, _, _ = await self._acquire_instance(
                        model_field,
                        excluded_instances=attempted_instances,
                    )
                except HTTPException as exc:
                    self.stats.total_errors += 1
                    if last_retryable_error and exc.status_code == 503:
                        raise HTTPException(
                            status_code=503,
                            detail=(
                                "All candidate QWN instances failed before "
                                f"response start: {last_retryable_error}"
                            ),
                        ) from exc
                    raise

                attempted_instances.add(instance.container_name)
                attempt_body = self._rewrite_model_in_body(body, instance)

                try:
                    started_at = time.perf_counter()
                    response = await self._client.post(
                        instance.chat_url,
                        content=attempt_body,
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code != 200:
                        if self._is_retryable_upstream_status(response.status_code):
                            last_retryable_error = (
                                f"{instance.container_name} returned "
                                f"{response.status_code}: {response.text}"
                            )
                            self.stats.total_failovers += 1
                            self._mark_instance_unhealthy(
                                instance, last_retryable_error
                            )
                            continue
                        self.stats.total_errors += 1
                        raise HTTPException(
                            status_code=response.status_code, detail=response.text
                        )

                    payload = orjson.loads(response.content)
                    usage = payload.get("usage", {})
                    self.stats.total_tokens += usage.get("total_tokens", 0)
                    instance.telemetry.record_success(
                        latency_ms=(time.perf_counter() - started_at) * 1000.0,
                        completion_tokens=usage.get("completion_tokens", 0),
                    )
                    return ChatCompletionResponse(
                        id=payload.get("id", ""),
                        object=payload.get("object", "chat.completion"),
                        created=payload.get("created", 0),
                        model=self._get_model_label(instance)
                        or payload.get("model", ""),
                        choices=[
                            ChatChoiceDelta(**choice)
                            for choice in payload.get("choices", [])
                        ],
                        usage=UsageInfo(**usage),
                    )
                except HTTPException:
                    raise
                except Exception as exc:
                    if self._is_retryable_upstream_exception(exc):
                        last_retryable_error = f"{instance.container_name}: {exc}"
                        self.stats.total_failovers += 1
                        self._mark_instance_unhealthy(instance, exc)
                        continue
                    self.stats.total_errors += 1
                    logger.warn(f"× Chat completion error: {exc}")
                    raise HTTPException(status_code=500, detail=str(exc))
                finally:
                    await self._release_instance(instance)
        except HTTPException:
            raise
        finally:
            self.stats.active_requests = max(0, self.stats.active_requests - 1)

    def run(self) -> None:
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")

    async def run_async(self) -> None:
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="info"
        )
        await uvicorn.Server(config).serve()


_CACHE_DIR = Path.home() / ".cache" / "tfmx"
_PID_FILE = _CACHE_DIR / "qwn_machine.pid"
_LOG_FILE = _CACHE_DIR / "qwn_machine.log"


class QWNMachineDaemon:
    def __init__(self, pid_file: Path = _PID_FILE, log_file: Path = _LOG_FILE):
        self.pid_file = pid_file
        self.log_file = log_file
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

    def get_pid(self) -> Optional[int]:
        if not self.pid_file.exists():
            return None
        try:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            self.pid_file.unlink(missing_ok=True)
            return None

    def is_running(self) -> bool:
        return self.get_pid() is not None

    def write_pid(self) -> None:
        self.pid_file.write_text(str(os.getpid()))

    def remove_pid(self) -> None:
        self.pid_file.unlink(missing_ok=True)

    @staticmethod
    def _normalize_background_argv(argv: list[str]) -> list[str]:
        normalized_args = list(argv[1:])
        if normalized_args and normalized_args[0] == "machine":
            normalized_args = normalized_args[1:]

        cleaned_args: list[str] = []
        for arg in normalized_args:
            if arg in {"-b", "--background"}:
                continue
            cleaned_args.append(arg)
        return cleaned_args

    def start_background(self, argv: list[str]) -> None:
        cmd = [sys.executable, "-m", "tfmx.qwns.machine"]
        for arg in self._normalize_background_argv(argv):
            cmd.append(arg)

        log_handle = open(self.log_file, "a")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "_QWN_MACHINE_DAEMON": "1"},
        )
        self.pid_file.write_text(str(proc.pid))
        logger.okay(f"[qwn_machine] Started in background (PID {proc.pid})")
        logger.mesg(f"  Log: {self.log_file}")
        logger.mesg(f"  PID: {self.pid_file}")

    def stop(self) -> bool:
        pid = self.get_pid()
        if pid is None:
            logger.mesg("[qwn_machine] No running daemon found")
            return False

        logger.mesg(f"[qwn_machine] Stopping daemon (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(100):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except ProcessLookupError:
                    break
            else:
                logger.warn(f"[qwn_machine] Force killing PID {pid}")
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warn(f"x Permission denied killing PID {pid}")
            return False

        self.remove_pid()
        logger.okay("[qwn_machine] Daemon stopped")
        return True

    def status(self) -> None:
        pid = self.get_pid()
        if pid is None:
            logger.mesg("[qwn_machine] Status: not running")
            return
        logger.okay(f"[qwn_machine] Status: running (PID {pid})")
        logger.mesg(f"  PID file: {self.pid_file}")
        logger.mesg(f"  Log file: {self.log_file}")
        if self.log_file.exists():
            size = self.log_file.stat().st_size
            if size < 1024:
                logger.mesg(f"  Log size: {size} B")
            elif size < 1024 * 1024:
                logger.mesg(f"  Log size: {size / 1024:.1f} KB")
            else:
                logger.mesg(f"  Log size: {size / 1024 / 1024:.1f} MB")

    def show_logs(self, follow: bool = False, tail: int = 50) -> None:
        if not self.log_file.exists():
            logger.mesg("[qwn_machine] No log file found")
            return
        if follow:
            try:
                subprocess.run(["tail", "-f", "-n", str(tail), str(self.log_file)])
            except KeyboardInterrupt:
                pass
            return

        result = subprocess.run(
            ["tail", "-n", str(tail), str(self.log_file)],
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        else:
            logger.mesg("[qwn_machine] Log file is empty")


CLI_EPILOG = """
Examples:
  qwn machine run
  qwn machine run -b
  qwn machine run -e http://localhost:27880,http://localhost:27881
  qwn machine status
  qwn machine logs -f
  qwn machine discover
  qwn machine health
"""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Run and manage the QWN machine proxy"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = CLI_EPILOG
    parser.add_argument(
        "action",
        nargs="?",
        choices=["run", "discover", "health", "stop", "restart", "status", "logs"],
        default=None,
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=PORT,
        help=f"Machine server port (default: {PORT})",
    )
    parser.add_argument(
        "-n",
        "--name-pattern",
        type=str,
        default=None,
        help="Regex to filter container names",
    )
    parser.add_argument(
        "-e",
        "--endpoints",
        type=str,
        default=None,
        help="Comma-separated backend endpoints",
    )
    parser.add_argument(
        "-t", "--timeout", type=float, default=120.0, help="Request timeout in seconds"
    )
    parser.add_argument(
        "-b", "--background", action="store_true", help="Run as background daemon"
    )
    parser.add_argument(
        "-f", "--follow", action="store_true", help="Follow logs in real time"
    )
    parser.add_argument("--tail", type=int, default=50, help="Log lines to show")


def discover_instances(args: argparse.Namespace) -> list[QWNInstance]:
    if args.endpoints:
        endpoints = [
            endpoint.strip()
            for endpoint in args.endpoints.split(",")
            if endpoint.strip()
        ]
        instances = QWNInstanceDiscovery.from_endpoints(endpoints)
        logger.okay(f"[qwn_machine] Using {len(instances)} manual endpoints")
    else:
        instances = QWNInstanceDiscovery.discover(args.name_pattern)
        logger.okay(f"[qwn_machine] Discovered {len(instances)} QWN instances")
    return instances


def log_instances(instances: list[QWNInstance], show_health: bool = False) -> None:
    if not instances:
        logger.warn("× No QWN instances found")
        return

    dash_len = 100
    logger.note("=" * dash_len)
    if show_health:
        logger.note(
            f"{'GPU':<6} {'CONTAINER':<35} {'ENDPOINT':<25} {'MODEL':<20} {'STATUS':<8}"
        )
    else:
        logger.note(f"{'GPU':<6} {'CONTAINER':<35} {'ENDPOINT':<25} {'MODEL':<20}")
    logger.note("-" * dash_len)

    for instance in instances:
        gpu_text = str(instance.gpu_id) if instance.gpu_id is not None else "?"
        model_text = instance.model_name or "?"
        if show_health:
            status = (
                logstr.okay("✓ healthy") if instance.healthy else logstr.erro("× sick")
            )
            logger.mesg(
                f"{gpu_text:<6} {instance.container_name:<35} {instance.endpoint:<25} {model_text:<20} {status:<8}"
            )
        else:
            logger.mesg(
                f"{gpu_text:<6} {instance.container_name:<35} {instance.endpoint:<25} {model_text:<20}"
            )

    logger.note("=" * dash_len)
    if show_health:
        healthy = sum(1 for instance in instances if instance.healthy)
        logger.mesg(f"[qwn_machine] Healthy: {healthy}/{len(instances)}")


async def check_health(instances: list[QWNInstance]) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for instance in instances:
            try:
                response = await client.get(instance.health_url)
                instance.healthy = response.status_code == 200
            except Exception:
                instance.healthy = False
    log_instances(instances, show_health=True)


def run_from_args(args: argparse.Namespace) -> None:
    if args.action is None:
        raise ValueError("Action is required")

    daemon = QWNMachineDaemon()

    if args.action == "stop":
        daemon.stop()
        return
    if args.action == "status":
        daemon.status()
        return
    if args.action == "logs":
        daemon.show_logs(follow=args.follow, tail=args.tail)
        return
    if args.action == "restart":
        daemon.stop()
        argv = [arg if arg != "restart" else "run" for arg in sys.argv]
        if "-b" not in argv and "--background" not in argv:
            argv.append("-b")
        daemon.start_background(argv)
        return

    instances = discover_instances(args)
    if args.action == "discover":
        log_instances(instances)
        return
    if args.action == "health":
        asyncio.run(check_health(instances))
        return

    if args.action == "run":
        if not instances:
            logger.warn(
                "× No QWN instances found. Use -e to specify endpoints manually."
            )
            return

        if args.background:
            if daemon.is_running():
                logger.warn(
                    f"[qwn_machine] Daemon already running (PID {daemon.get_pid()}). Use 'restart' to replace it."
                )
                return
            daemon.start_background(sys.argv)
            return

        is_daemon = os.environ.get("_QWN_MACHINE_DAEMON") == "1"
        if is_daemon:
            daemon.write_pid()

        server = QWNMachineServer(
            instances=instances, port=args.port, timeout=args.timeout
        )
        try:
            server.run()
        finally:
            if is_daemon:
                daemon.remove_pid()
        return

    raise ValueError(f"Unknown machine action: {args.action}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        return
    run_from_args(args)


if __name__ == "__main__":
    main()
