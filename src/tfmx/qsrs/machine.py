"""QSR machine proxy server and daemon management."""

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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from pathlib import Path
from pydantic import BaseModel, Field
from tclogger import logger, logstr
from typing import Callable, Literal, Optional, Union
from urllib.parse import urlsplit
from webu import setup_swagger_ui

from ..utils.service_bootstrap import docker_status_to_health
from ..utils.service_bootstrap import ensure_backend_instances
from ..utils.service_bootstrap import handle_port_conflicts
from ..utils.service_bootstrap import wait_for_healthy_http_endpoints
from .client import _build_transcription_multipart_fields
from .compose import GPU_LAYOUT_UNIFORM, MACHINE_PORT
from .compose import MAX_MODEL_LEN, MAX_NUM_SEQS
from .compose import MODEL_NAME as COMPOSE_MODEL_NAME
from .compose import QSRComposer, SERVER_PORT as COMPOSE_SERVER_PORT
from .compose import SLEEP_CONTROL_TIMEOUT_SEC, SLEEP_WAKE_POLL_INTERVAL_SEC
from .compose import get_backend_sleep_state, set_backend_sleep_states
from .compose import get_display_shortcut, get_model_api_aliases
from .compose import get_model_shortcut, normalize_model_key, parse_gpu_configs
from .router import InstanceDescriptor, QSRRouter


PORT = MACHINE_PORT
MAX_CONCURRENT = MAX_NUM_SEQS
DEFAULT_MAX_TOKENS = MAX_MODEL_LEN
DEFAULT_NAME_PATTERN = r"qsr[-_]"
HEALTH_REFRESH_INTERVAL_SEC = 5.0
HEALTH_REQUEST_TIMEOUT_SEC = 5.0
MODEL_DISCOVERY_TIMEOUT_SEC = 10.0
INSTANCE_ACQUIRE_TIMEOUT_SEC = 5.0
BACKEND_STARTUP_TIMEOUT_SEC = 300.0
BACKEND_STARTUP_POLL_INTERVAL_SEC = 1.0
BACKEND_DISCOVERY_SETTLE_SEC = 2.0
SCHEDULER_ALGORITHM = "least_active_idle"


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")


class AudioURL(BaseModel):
    url: str = Field(..., description="Audio URL or base64 data URI")


class AudioContent(BaseModel):
    type: Literal["audio_url"] = "audio_url"
    audio_url: AudioURL


ContentPart = Union[TextContent, AudioContent]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Message role"
    )
    content: Union[str, list[ContentPart]] = Field(
        ...,
        description=(
            "Message content: plain text string, or a list of text/audio_url parts "
            "for ASR prompts"
        ),
    )


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str = Field(default="", description="Model label or shortcut")
    messages: list[ChatMessage] = Field(..., description="OpenAI chat messages")
    max_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Maximum tokens to generate; omitted by default so OpenAI-compatible "
            "clients do not accidentally request the full model context length"
        ),
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
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
    sleeping: bool = Field(False, description="Whether instance is in vLLM sleep mode")
    latency_ms: float | None = Field(None, description="Last health check latency")
    active_requests: int = Field(0, description="Current active requests")
    model_label: str = Field(default="", description="Model label")
    gpu_id: int | None = Field(None, description="GPU ID")


class HealthResponse(BaseModel):
    status: str = Field(...)
    healthy: int = Field(...)
    total: int = Field(...)
    instances: list[InstanceHealthDetail] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str = Field(...)
    object: str = Field(default="model")
    created: int = Field(default=0)
    owned_by: str = Field(default="qsr-machine")


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
    sleeping: bool = Field(False)
    model_name: str = Field(default="")
    model_label: str = Field(default="")
    active_requests: int = Field(0)
    available_slots: int = Field(0)
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
    acquire_timeout_sec: float = Field(default=INSTANCE_ACQUIRE_TIMEOUT_SEC)
    last_health_refresh_age_sec: float | None = Field(None)


class InfoResponse(BaseModel):
    port: int
    instances: list[InstanceInfo] = Field(default_factory=list)
    stats: MachineStats = Field(default_factory=MachineStats)
    available_models: list[str] = Field(default_factory=list)
    scheduler: SchedulerInfo = Field(default_factory=SchedulerInfo)


@dataclass
class QSRInstance:
    container_name: str
    host: str
    port: int
    gpu_id: Optional[int] = None
    healthy: bool = False
    sleeping: bool = False
    sleep_mode_supported: Optional[bool] = None
    docker_status: str = ""
    docker_health: bool | None = None
    _active_requests: int = 0
    model_name: str = ""
    _latency_ms: float = 0.0
    last_error: str = ""

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def chat_url(self) -> str:
        return f"{self.endpoint}/v1/chat/completions"

    @property
    def transcription_url(self) -> str:
        return f"{self.endpoint}/v1/audio/transcriptions"

    @property
    def health_url(self) -> str:
        return f"{self.endpoint}/health"

    @property
    def models_url(self) -> str:
        return f"{self.endpoint}/v1/models"

    @property
    def sleep_state_url(self) -> str:
        return f"{self.endpoint}/is_sleeping"

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
        return f"QSRInstance({status} {self.container_name} @ {self.endpoint}, {gpu_text}{model_text})"

    def to_info(self, model_label: str = "") -> InstanceInfo:
        return InstanceInfo(
            name=self.container_name,
            endpoint=self.endpoint,
            gpu_id=self.gpu_id,
            healthy=self.healthy,
            sleeping=self.sleeping,
            model_name=self.model_name,
            model_label=model_label or self.model_name,
            active_requests=self._active_requests,
            available_slots=self.available_slots,
            scheduler=InstanceSchedulerInfo(
                recent_requests=self._active_requests,
                recent_successes=0,
                recent_failures=0,
                latency_ema_ms=round(self._latency_ms, 2) if self._latency_ms else None,
                last_error=self.last_error,
            ),
        )


def _apply_qsr_docker_health(instance: QSRInstance) -> bool | None:
    if instance.docker_health is None:
        return None
    instance._latency_ms = 0.0
    if not instance.docker_health:
        instance.healthy = False
        instance.sleeping = False
        return False

    sleeping_state = get_backend_sleep_state(instance.endpoint)
    instance.sleeping = bool(sleeping_state)
    if instance.sleeping:
        instance.sleep_mode_supported = True
    instance.healthy = not instance.sleeping
    return instance.healthy


def _wake_sleeping_backends(
    endpoints: list[str],
    *,
    timeout_sec: float,
    poll_interval_sec: float,
    label: str,
) -> bool:
    normalized = [endpoint.rstrip("/") for endpoint in endpoints if endpoint]
    if not normalized:
        return False

    successes: list[str] = []
    with httpx.Client(timeout=httpx.Timeout(SLEEP_CONTROL_TIMEOUT_SEC)) as client:
        for endpoint in normalized:
            try:
                response = client.post(f"{endpoint}/wake_up")
            except Exception as exc:
                logger.warn(f"{label} Wake request failed for {endpoint}: {exc}")
                continue

            if response.status_code == 200:
                successes.append(endpoint)
            elif response.status_code == 404:
                logger.warn(
                    f"{label} Wake endpoint unavailable for {endpoint}; redeploy with --enable-sleep-mode"
                )
            else:
                logger.warn(
                    f"{label} Wake request failed for {endpoint}: HTTP {response.status_code}"
                )

    if not successes:
        return False

    set_backend_sleep_states(successes, sleeping=False)
    return wait_for_healthy_http_endpoints(
        successes,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
        request_timeout_sec=HEALTH_REQUEST_TIMEOUT_SEC,
        label=label,
    )


@dataclass
class QSRStatsData:
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


class QSRInstanceDiscovery:
    @staticmethod
    def discover(name_pattern: Optional[str] = None) -> list[QSRInstance]:
        try:
            result = subprocess.run(
                "docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}|{{.Status}}'",
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
            instances: list[QSRInstance] = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                container_name, _image, ports = parts[0], parts[1], parts[2]
                status = parts[3].strip() if len(parts) > 3 else ""
                if not re.search(pattern, container_name):
                    continue
                host_port = QSRInstanceDiscovery._extract_host_port(
                    ports, container_name
                )
                if host_port is None:
                    continue
                gpu_id = QSRInstanceDiscovery._extract_gpu_id(container_name)
                instances.append(
                    QSRInstance(
                        container_name=container_name,
                        host="localhost",
                        port=host_port,
                        gpu_id=gpu_id,
                        docker_status=status,
                        docker_health=docker_status_to_health(status),
                    )
                )
            instances.sort(
                key=lambda instance: (
                    instance.gpu_id if instance.gpu_id is not None else 999,
                    instance.container_name,
                )
            )
            return instances
        except Exception as exc:
            logger.warn(f"× Failed to discover QSR instances: {exc}")
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
    def from_endpoints(endpoints: list[str]) -> list[QSRInstance]:
        instances: list[QSRInstance] = []
        for index, endpoint in enumerate(endpoints):
            endpoint = endpoint.strip()
            if not endpoint:
                continue
            if re.fullmatch(r"\d+", endpoint):
                host = "localhost"
                port = int(endpoint)
            else:
                parsed = urlsplit(
                    endpoint if "://" in endpoint else f"http://{endpoint}"
                )
                host = parsed.hostname or "localhost"
                port = parsed.port
                if port is None and parsed.path and parsed.path.isdigit():
                    port = int(parsed.path)
            if port is None:
                continue
            instances.append(
                QSRInstance(
                    container_name=f"manual-{index}",
                    host=host,
                    port=port,
                    gpu_id=index,
                )
            )
        return instances


class QSRMachineServer:
    def __init__(
        self,
        instances: list[QSRInstance],
        port: int = PORT,
        timeout: float = 120.0,
        discover_instances_fn: Callable[[], list[QSRInstance]] | None = None,
    ):
        self.instances = sorted(
            instances,
            key=lambda instance: (
                instance.gpu_id if instance.gpu_id is not None else 999,
                instance.container_name,
            ),
        )
        self.port = port
        self.timeout = timeout
        self._discover_instances_fn = discover_instances_fn
        self.stats = QSRStatsData()
        self._client: Optional[httpx.AsyncClient] = None
        self._transcription_client: Optional[httpx.Client] = None
        self._health_task: Optional[asyncio.Task] = None
        self._capacity_cond: Optional[asyncio.Condition] = None
        self._idle_tiebreak_counter: int = 0
        self._last_health_refresh_monotonic: float = 0.0
        self.router = QSRRouter()
        self._build_router()
        self.app = self._create_app()

    def _build_router(self) -> None:
        self.router = QSRRouter()
        for instance in self.instances:
            self.router.register(
                InstanceDescriptor(
                    model_name=instance.model_name or COMPOSE_MODEL_NAME,
                    endpoint=instance.endpoint,
                    gpu_id=instance.gpu_id,
                    instance_id=instance.container_name,
                    healthy=instance.healthy,
                )
            )

    def _get_default_route_descriptor(self) -> InstanceDescriptor | None:
        descriptor = self.router.route()
        if descriptor is not None:
            return descriptor
        if not self.instances:
            return None
        instance = self.instances[0]
        return InstanceDescriptor(
            model_name=instance.model_name or COMPOSE_MODEL_NAME,
            endpoint=instance.endpoint,
            gpu_id=instance.gpu_id,
            instance_id=instance.container_name,
            healthy=instance.healthy,
        )

    def _get_default_model_field(self) -> str:
        descriptor = self._get_default_route_descriptor()
        if descriptor is None:
            return ""
        return descriptor.label or descriptor.model_name or ""

    def _resolve_requested_model_field(self, model_field: str = "") -> str:
        model_field = (model_field or "").strip()
        if model_field:
            return model_field
        return self._get_default_model_field()

    def get_healthy_instances(self) -> list[QSRInstance]:
        return [instance for instance in self.instances if instance.healthy]

    def _get_model_label(self, instance: QSRInstance) -> str:
        shortcut = get_model_shortcut(instance.model_name or COMPOSE_MODEL_NAME)
        display = get_display_shortcut(shortcut) if shortcut else ""
        return display or shortcut or instance.model_name or COMPOSE_MODEL_NAME

    def _get_public_model_ids(self, instance: QSRInstance) -> list[str]:
        public_ids: list[str] = []
        model_name = instance.model_name or COMPOSE_MODEL_NAME
        for alias in get_model_api_aliases(model_name):
            if alias and alias not in public_ids:
                public_ids.append(alias)
        shortcut = get_model_shortcut(model_name)
        if shortcut and shortcut not in public_ids:
            public_ids.append(shortcut)
        if model_name and model_name not in public_ids:
            public_ids.append(model_name)
        return public_ids

    @staticmethod
    def _instance_identity(instance: QSRInstance) -> str:
        return instance.container_name or instance.endpoint

    def _merge_discovered_instances(
        self,
        discovered: list[QSRInstance],
    ) -> tuple[bool, list[QSRInstance]]:
        existing_by_identity = {
            self._instance_identity(instance): instance for instance in self.instances
        }
        discovered_identities: set[str] = set()
        changed = False
        added_instances: list[QSRInstance] = []

        for discovered_instance in discovered:
            identity = self._instance_identity(discovered_instance)
            discovered_identities.add(identity)
            existing = existing_by_identity.get(identity)
            if existing is None:
                self.instances.append(discovered_instance)
                existing_by_identity[identity] = discovered_instance
                added_instances.append(discovered_instance)
                changed = True
                continue

            if (
                existing.host != discovered_instance.host
                or existing.port != discovered_instance.port
                or existing.gpu_id != discovered_instance.gpu_id
                or existing.docker_status != discovered_instance.docker_status
                or existing.docker_health != discovered_instance.docker_health
            ):
                existing.host = discovered_instance.host
                existing.port = discovered_instance.port
                existing.gpu_id = discovered_instance.gpu_id
                existing.docker_status = discovered_instance.docker_status
                existing.docker_health = discovered_instance.docker_health
                changed = True

        for identity, existing in existing_by_identity.items():
            if identity in discovered_identities:
                continue
            if (
                existing.docker_status != "missing"
                or existing.docker_health is not False
            ):
                existing.docker_status = "missing"
                existing.docker_health = False
                existing.healthy = False
                existing.sleeping = False
                changed = True

        if changed:
            self.instances.sort(
                key=lambda instance: (
                    instance.gpu_id if instance.gpu_id is not None else 999,
                    instance.container_name,
                )
            )
        return changed, added_instances

    async def _refresh_instances_from_discovery(self) -> None:
        if self._discover_instances_fn is None:
            return
        discovered = await asyncio.to_thread(self._discover_instances_fn)
        changed, added_instances = self._merge_discovered_instances(discovered)
        if not changed:
            return
        for instance in added_instances:
            logger.okay(
                f"[qsr_machine] Added backend instance: {instance.container_name} @ {instance.endpoint}"
            )
        self._build_router()
        await self._notify_capacity_changed()

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

    async def _check_instance_health(self, instance: QSRInstance) -> bool:
        quiet_health = _apply_qsr_docker_health(instance)
        if quiet_health is not None:
            return quiet_health
        try:
            started_at = time.perf_counter()
            response = await self._client.get(
                instance.health_url,
                timeout=httpx.Timeout(HEALTH_REQUEST_TIMEOUT_SEC),
            )
            instance.healthy = response.status_code == 200
            instance.sleeping = False
            if instance.healthy:
                try:
                    sleep_response = await self._client.get(
                        instance.sleep_state_url,
                        timeout=httpx.Timeout(HEALTH_REQUEST_TIMEOUT_SEC),
                    )
                    if sleep_response.status_code == 200:
                        instance.sleep_mode_supported = True
                        payload = sleep_response.json()
                        instance.sleeping = bool(payload.get("is_sleeping", False))
                        if instance.sleeping:
                            instance.healthy = False
                    elif sleep_response.status_code == 404:
                        instance.sleep_mode_supported = False
                except Exception:
                    if instance.sleep_mode_supported is None:
                        instance.sleep_mode_supported = False
            instance._latency_ms = (
                (time.perf_counter() - started_at) * 1000 if instance.healthy else 0.0
            )
            return instance.healthy
        except Exception:
            instance.healthy = False
            instance.sleeping = False
            instance._latency_ms = 0.0
            return False

    async def _discover_instance_model(self, instance: QSRInstance) -> None:
        for url in (instance.models_url, f"{instance.endpoint}/models"):
            try:
                response = await self._client.get(
                    url,
                    timeout=httpx.Timeout(MODEL_DISCOVERY_TIMEOUT_SEC),
                )
                if response.status_code != 200:
                    continue
                payload = response.json()
                models = payload.get("data", [])
                if not models:
                    continue
                instance.model_name = models[0].get("id", "") or COMPOSE_MODEL_NAME
                return
            except Exception:
                continue
        if not instance.model_name:
            instance.model_name = COMPOSE_MODEL_NAME

    async def _refresh_health(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        await self._refresh_instances_from_discovery()
        changed = False
        for instance in self.instances:
            was_healthy = instance.healthy
            await self._check_instance_health(instance)
            if instance.healthy and not instance.model_name:
                await self._discover_instance_model(instance)
            if instance.healthy != was_healthy:
                changed = True
        self._last_health_refresh_monotonic = time.monotonic()
        if changed:
            self._build_router()
            await self._notify_capacity_changed()

    async def _health_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(HEALTH_REFRESH_INTERVAL_SEC)
                await self._refresh_health()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warn(f"[qsr_machine] Health refresh failed: {exc}")

    def _get_candidate_instances(
        self,
        model_field: str = "",
        excluded_instances: set[str] | None = None,
        require_idle: bool = False,
    ) -> list[QSRInstance]:
        excluded_instances = excluded_instances or set()
        candidates = [
            instance
            for instance in self.instances
            if instance.healthy and instance.container_name not in excluded_instances
        ]
        if model_field:
            descriptors = [
                descriptor
                for descriptor in self.router.find_instances(model_field)
                if descriptor.healthy
            ]
            endpoints = {descriptor.endpoint for descriptor in descriptors}
            candidates = [
                instance for instance in candidates if instance.endpoint in endpoints
            ]
        if require_idle:
            candidates = [instance for instance in candidates if instance.is_idle]
        return candidates

    def _select_idle_instance(
        self,
        model_field: str = "",
        excluded_instances: set[str] | None = None,
    ) -> tuple[QSRInstance | None, bool]:
        candidates = self._get_candidate_instances(
            model_field=model_field,
            excluded_instances=excluded_instances,
            require_idle=False,
        )
        if not candidates:
            return None, False
        idle_candidates = [instance for instance in candidates if instance.is_idle]
        if not idle_candidates:
            return None, True
        idle_candidates.sort(
            key=lambda instance: (
                instance._active_requests,
                instance.gpu_id if instance.gpu_id is not None else 999,
                instance.container_name,
            )
        )
        lowest_active = idle_candidates[0]._active_requests
        tied_candidates = [
            instance
            for instance in idle_candidates
            if instance._active_requests == lowest_active
        ]
        index = self._idle_tiebreak_counter % len(tied_candidates)
        self._idle_tiebreak_counter += 1
        return tied_candidates[index], True

    def _reserve_idle_instance_locked(
        self,
        model_field: str = "",
        excluded_instances: set[str] | None = None,
    ) -> tuple[QSRInstance | None, bool]:
        instance, has_candidates = self._select_idle_instance(
            model_field=model_field,
            excluded_instances=excluded_instances,
        )
        if instance is None:
            return None, has_candidates
        instance._active_requests += 1
        self.stats.requests_per_instance[instance.container_name] = (
            self.stats.requests_per_instance.get(instance.container_name, 0) + 1
        )
        self.stats.active_requests += 1
        return instance, True

    async def _acquire_instance(
        self,
        model_field: str = "",
        excluded_instances: set[str] | None = None,
    ) -> QSRInstance:
        excluded_instances = excluded_instances or set()
        self._ensure_scheduler_primitives()
        waited_sec = 0.0
        deadline = time.perf_counter() + INSTANCE_ACQUIRE_TIMEOUT_SEC

        async with self._capacity_cond:
            while True:
                instance, has_candidates = self._reserve_idle_instance_locked(
                    model_field=model_field,
                    excluded_instances=excluded_instances,
                )
                if instance is not None:
                    if waited_sec > 0:
                        self.stats.record_wait(waited_sec * 1000.0)
                    return instance

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    if waited_sec > 0:
                        self.stats.record_wait(waited_sec * 1000.0)
                    detail = "All QSR instances are busy"
                    if model_field:
                        detail = f"All QSR instances for requested model '{model_field}' are busy"
                    raise HTTPException(status_code=503, detail=detail)

                if not has_candidates:
                    if waited_sec > 0:
                        self.stats.record_wait(waited_sec * 1000.0)
                    detail = "No available QSR instances"
                    if model_field:
                        detail = f"No available QSR instances for requested model '{model_field}'"
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

    async def _release_instance(self, instance: QSRInstance) -> None:
        instance._active_requests = max(0, instance._active_requests - 1)
        self.stats.active_requests = max(0, self.stats.active_requests - 1)
        await self._notify_capacity_changed()

    def _mark_instance_unhealthy(
        self, instance: QSRInstance, reason: Exception | str
    ) -> None:
        reason_text = str(reason)
        if instance.healthy:
            logger.warn(
                f"× Marking {instance.container_name} unhealthy for failover: {reason_text}"
            )
        instance.healthy = False
        instance.last_error = reason_text
        instance._latency_ms = 0.0
        self._build_router()
        self._notify_capacity_changed_soon()

    def _rewrite_model_in_body(self, body: bytes, instance: QSRInstance) -> bytes:
        try:
            payload = orjson.loads(body)
            if instance.model_name:
                payload["model"] = instance.model_name
            elif not payload.get("model"):
                payload["model"] = self._get_default_model_field() or COMPOSE_MODEL_NAME
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

    async def _read_error_detail(self, response: httpx.Response) -> object:
        try:
            payload = response.json()
            if isinstance(payload, dict) and "error" in payload:
                return payload["error"]
            return payload
        except Exception:
            text = response.text.strip()
            return text or response.reason_phrase

    def _post_transcription_sync(
        self,
        url: str,
        files: list[tuple[str, object]],
    ) -> tuple[int, dict[str, str], bytes]:
        if self._transcription_client is not None:
            response = self._transcription_client.post(url, files=files)
            return response.status_code, dict(response.headers), response.content

        with httpx.Client(timeout=httpx.Timeout(self.timeout)) as client:
            response = client.post(url, files=files)
            return response.status_code, dict(response.headers), response.content

    async def _forward_chat(
        self,
        body: bytes,
        requested_model_field: str = "",
    ) -> ChatCompletionResponse:
        excluded_instances: set[str] = set()
        last_exc: HTTPException | None = None

        while True:
            instance = await self._acquire_instance(
                model_field=requested_model_field,
                excluded_instances=excluded_instances,
            )
            try:
                response = await self._client.post(
                    instance.chat_url,
                    content=self._rewrite_model_in_body(body, instance),
                    headers={"content-type": "application/json"},
                    timeout=httpx.Timeout(self.timeout),
                )
                if response.status_code >= 400:
                    detail = await self._read_error_detail(response)
                    if self._is_retryable_upstream_status(response.status_code):
                        excluded_instances.add(instance.container_name)
                        self.stats.total_failovers += 1
                        self._mark_instance_unhealthy(instance, detail)
                        last_exc = HTTPException(
                            status_code=response.status_code,
                            detail=detail,
                        )
                        continue
                    self.stats.total_errors += 1
                    raise HTTPException(status_code=response.status_code, detail=detail)

                payload = response.json()
                usage = payload.get("usage") or {}
                self.stats.total_requests += 1
                self.stats.total_tokens += int(usage.get("total_tokens", 0) or 0)
                return ChatCompletionResponse.model_validate(payload)
            except HTTPException:
                raise
            except Exception as exc:
                if self._is_retryable_upstream_exception(exc):
                    excluded_instances.add(instance.container_name)
                    self.stats.total_failovers += 1
                    self._mark_instance_unhealthy(instance, exc)
                    last_exc = HTTPException(status_code=502, detail=str(exc))
                    continue
                self.stats.total_errors += 1
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            finally:
                await self._release_instance(instance)

            if last_exc is not None:
                self.stats.total_errors += 1
                raise last_exc

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

    async def _forward_stream(
        self,
        body: bytes,
        requested_model_field: str = "",
    ) -> StreamingResponse:
        async def stream_generator():
            excluded_instances: set[str] = set()
            while True:
                try:
                    instance = await self._acquire_instance(
                        model_field=requested_model_field,
                        excluded_instances=excluded_instances,
                    )
                except HTTPException as exc:
                    self.stats.total_errors += 1
                    yield self._build_stream_error_chunk(
                        str(exc.detail), exc.status_code, "proxy_error"
                    )
                    return

                try:
                    async with self._client.stream(
                        "POST",
                        instance.chat_url,
                        content=self._rewrite_model_in_body(body, instance),
                        headers={"content-type": "application/json"},
                        timeout=httpx.Timeout(self.timeout),
                    ) as response:
                        if response.status_code >= 400:
                            detail = await self._read_error_detail(response)
                            if self._is_retryable_upstream_status(response.status_code):
                                excluded_instances.add(instance.container_name)
                                self.stats.total_failovers += 1
                                self._mark_instance_unhealthy(instance, detail)
                                continue
                            self.stats.total_errors += 1
                            yield self._build_stream_error_chunk(
                                str(detail),
                                response.status_code,
                                "upstream_error",
                            )
                            return

                        self.stats.total_requests += 1
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
                        return
                except Exception as exc:
                    if self._is_retryable_upstream_exception(exc):
                        excluded_instances.add(instance.container_name)
                        self.stats.total_failovers += 1
                        self._mark_instance_unhealthy(instance, exc)
                        continue
                    self.stats.total_errors += 1
                    yield self._build_stream_error_chunk(
                        str(exc),
                        502,
                        "proxy_error",
                    )
                    return
                finally:
                    await self._release_instance(instance)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    async def _forward_transcription(
        self,
        *,
        filename: str,
        payload: bytes,
        content_type: str,
        model: str = "",
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        timestamp_granularities: list[str] | None = None,
    ) -> Response:
        requested_model_field = self._resolve_requested_model_field(model)
        excluded_instances: set[str] = set()

        while True:
            instance = await self._acquire_instance(
                model_field=requested_model_field,
                excluded_instances=excluded_instances,
            )
            try:
                upstream_model = (
                    instance.model_name or requested_model_field or COMPOSE_MODEL_NAME
                )
                status_code, headers, content = await asyncio.to_thread(
                    self._post_transcription_sync,
                    instance.transcription_url,
                    _build_transcription_multipart_fields(
                        filename=filename,
                        payload=payload,
                        mime_type=content_type,
                        model=upstream_model,
                        language=language,
                        prompt=prompt,
                        response_format=response_format,
                        temperature=temperature,
                        timestamp_granularities=timestamp_granularities,
                    ),
                )
                response = httpx.Response(
                    status_code,
                    headers=headers,
                    content=content,
                )
                if response.status_code >= 400:
                    detail = await self._read_error_detail(response)
                    if self._is_retryable_upstream_status(response.status_code):
                        excluded_instances.add(instance.container_name)
                        self.stats.total_failovers += 1
                        self._mark_instance_unhealthy(instance, detail)
                        continue
                    self.stats.total_errors += 1
                    raise HTTPException(status_code=response.status_code, detail=detail)

                self.stats.total_requests += 1
                content_type_header = response.headers.get("content-type", "")
                if "application/json" in content_type_header.lower():
                    return JSONResponse(content=response.json())
                media_type = content_type_header or "text/plain; charset=utf-8"
                return Response(content=response.content, media_type=media_type)
            except HTTPException:
                raise
            except Exception as exc:
                if self._is_retryable_upstream_exception(exc):
                    excluded_instances.add(instance.container_name)
                    self.stats.total_failovers += 1
                    self._mark_instance_unhealthy(instance, exc)
                    continue
                self.stats.total_errors += 1
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            finally:
                await self._release_instance(instance)

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
                    sleeping=instance.sleeping,
                    latency_ms=(
                        round(instance._latency_ms, 2)
                        if instance._latency_ms > 0
                        else None
                    ),
                    active_requests=instance._active_requests,
                    model_label=self._get_model_label(instance),
                    gpu_id=instance.gpu_id,
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
            for model_id in self._get_public_model_ids(instance):
                if model_id in seen:
                    continue
                seen.add(model_id)
                data.append(
                    ModelInfo(id=model_id, created=created, owned_by="qsr-machine")
                )
        return ModelsResponse(data=data)

    async def info(self) -> InfoResponse:
        now = time.monotonic()
        return InfoResponse(
            port=self.port,
            instances=[
                instance.to_info(model_label=self._get_model_label(instance))
                for instance in self.instances
            ],
            stats=self.stats.to_model(),
            available_models=self.router.get_available_models(),
            scheduler=SchedulerInfo(
                algorithm=SCHEDULER_ALGORITHM,
                acquire_timeout_sec=INSTANCE_ACQUIRE_TIMEOUT_SEC,
                last_health_refresh_age_sec=(
                    round(now - self._last_health_refresh_monotonic, 2)
                    if self._last_health_refresh_monotonic > 0
                    else None
                ),
            ),
        )

    async def chat_completions(self, request: ChatCompletionRequest):
        requested_model = self._resolve_requested_model_field(request.model)
        body = orjson.dumps(request.model_dump(exclude_none=True))
        if request.stream:
            return await self._forward_stream(body, requested_model)
        return await self._forward_chat(body, requested_model)

    async def chat_form(
        self,
        text: str = Form(..., description="Prompt text"),
        system_prompt: str = Form(default="", description="Optional system prompt"),
        model: str = Form(default="", description="Model label"),
        max_tokens: Optional[int] = Form(
            default=None,
            description=(
                "Maximum tokens; omitted by default so prompts still leave room "
                "for input context"
            ),
        ),
        temperature: float = Form(default=0.0, description="Temperature"),
        top_p: float = Form(default=1.0, description="Top-p"),
    ) -> ChatCompletionResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        body = orjson.dumps(payload)
        return await self._forward_chat(
            body,
            self._resolve_requested_model_field(model),
        )

    async def audio_transcriptions(
        self,
        file: UploadFile = File(...),
        model: str = Form(default=""),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float | None = Form(default=None),
        timestamp_granularities: list[str] | None = Form(default=None),
    ) -> Response:
        payload = await file.read()
        return await self._forward_transcription(
            filename=file.filename or "audio.wav",
            payload=payload,
            content_type=file.content_type or "audio/wav",
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
        )

    def _build_metrics_payload(self) -> str:
        stats_model = self.stats.to_model()
        healthy_instances = self.get_healthy_instances()
        lines = [
            "# HELP qsr_machine_up Whether the qsr machine process is serving requests.",
            "# TYPE qsr_machine_up gauge",
            "qsr_machine_up 1",
            "# HELP qsr_machine_instances_total Total number of discovered qsr backend instances.",
            "# TYPE qsr_machine_instances_total gauge",
            f"qsr_machine_instances_total {len(self.instances)}",
            "# HELP qsr_machine_instances_healthy_total Number of currently healthy qsr backend instances.",
            "# TYPE qsr_machine_instances_healthy_total gauge",
            f"qsr_machine_instances_healthy_total {len(healthy_instances)}",
            "# HELP qsr_machine_requests_total Total requests handled by qsr machine.",
            "# TYPE qsr_machine_requests_total counter",
            f"qsr_machine_requests_total {stats_model.total_requests}",
            "# HELP qsr_machine_errors_total Total proxy-visible qsr machine errors.",
            "# TYPE qsr_machine_errors_total counter",
            f"qsr_machine_errors_total {stats_model.total_errors}",
            "# HELP qsr_machine_failovers_total Total pre-response failovers performed by qsr machine.",
            "# TYPE qsr_machine_failovers_total counter",
            f"qsr_machine_failovers_total {stats_model.total_failovers}",
            "# HELP qsr_machine_active_requests Number of requests currently active in qsr machine.",
            "# TYPE qsr_machine_active_requests gauge",
            f"qsr_machine_active_requests {stats_model.active_requests}",
        ]
        return "\n".join(lines) + "\n"

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        self._transcription_client = httpx.Client(timeout=httpx.Timeout(self.timeout))
        await self._refresh_health()
        self._health_task = asyncio.create_task(self._health_loop())
        try:
            yield
        finally:
            if self._health_task is not None:
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            if self._transcription_client is not None:
                self._transcription_client.close()
                self._transcription_client = None

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="QSR Machine",
            description=(
                "Load-balanced proxy for Qwen3-ASR vLLM instances.\n\n"
                "- POST /v1/chat/completions: OpenAI-compatible chat API\n"
                "- POST /chat/completions: OpenAI-compatible chat API alias\n"
                "- POST /v1/audio/transcriptions: OpenAI-compatible transcription API\n"
                "- POST /audio/transcriptions: OpenAI-compatible transcription API alias\n"
                "- GET /v1/models: list available models\n"
                "- GET /models: list available models alias\n"
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
                409: "conflict_error",
                422: "invalid_request_error",
                429: "rate_limit_error",
                500: "server_error",
                502: "upstream_error",
                503: "service_unavailable",
                504: "timeout_error",
            }
            payload = {
                "error": {
                    "message": str(detail),
                    "type": error_types.get(exc.status_code, "server_error"),
                    "code": exc.status_code,
                }
            }
            return JSONResponse(status_code=exc.status_code, content=payload)

        @app.get("/health", response_model=HealthResponse)
        async def health_route():
            return await self.health()

        @app.get("/v1/models", response_model=ModelsResponse)
        async def models_route():
            return await self.models()

        @app.get("/models", response_model=ModelsResponse)
        async def models_alias_route():
            return await self.models()

        @app.get("/info", response_model=InfoResponse)
        async def info_route():
            return await self.info()

        @app.get("/metrics")
        async def metrics_route():
            return PlainTextResponse(self._build_metrics_payload())

        @app.post("/v1/chat/completions")
        async def chat_completions_route(request: ChatCompletionRequest):
            return await self.chat_completions(request)

        @app.post("/chat/completions")
        async def chat_completions_alias_route(request: ChatCompletionRequest):
            return await self.chat_completions(request)

        @app.post("/chat")
        async def chat_form_route(
            text: str = Form(...),
            system_prompt: str = Form(default=""),
            model: str = Form(default=""),
            max_tokens: Optional[int] = Form(default=None),
            temperature: float = Form(default=0.0),
            top_p: float = Form(default=1.0),
        ):
            return await self.chat_form(
                text=text,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        @app.post("/v1/audio/transcriptions")
        async def audio_transcriptions_route(
            file: UploadFile = File(...),
            model: str = Form(default=""),
            language: str | None = Form(default=None),
            prompt: str | None = Form(default=None),
            response_format: str = Form(default="json"),
            temperature: float | None = Form(default=None),
            timestamp_granularities: list[str] | None = Form(default=None),
        ):
            return await self.audio_transcriptions(
                file=file,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
            )

        @app.post("/audio/transcriptions")
        async def audio_transcriptions_alias_route(
            file: UploadFile = File(...),
            model: str = Form(default=""),
            language: str | None = Form(default=None),
            prompt: str | None = Form(default=None),
            response_format: str = Form(default="json"),
            temperature: float | None = Form(default=None),
            timestamp_granularities: list[str] | None = Form(default=None),
        ):
            return await self.audio_transcriptions(
                file=file,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
            )

        return app

    def run(self) -> None:
        logger.okay(f"[qsr_machine] Starting on 0.0.0.0:{self.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")


_CACHE_DIR = Path.home() / ".cache" / "tfmx"
_PID_FILE = _CACHE_DIR / "qsr_machine.pid"
_LOG_FILE = _CACHE_DIR / "qsr_machine.log"


class QSRMachineDaemon:
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
        cmd = [sys.executable, "-m", "tfmx.qsrs.machine"]
        for arg in self._normalize_background_argv(argv):
            cmd.append(arg)

        log_handle = open(self.log_file, "a")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "_QSR_MACHINE_DAEMON": "1"},
        )
        self.pid_file.write_text(str(proc.pid))
        logger.okay(f"[qsr_machine] Started in background (PID {proc.pid})")
        logger.mesg(f"  Log: {self.log_file}")
        logger.mesg(f"  PID: {self.pid_file}")

    def stop(self) -> bool:
        pid = self.get_pid()
        if pid is None:
            logger.mesg("[qsr_machine] No running daemon found")
            return False

        logger.mesg(f"[qsr_machine] Stopping daemon (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(100):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except ProcessLookupError:
                    break
            else:
                logger.warn(f"[qsr_machine] Force killing PID {pid}")
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warn(f"x Permission denied killing PID {pid}")
            return False

        self.remove_pid()
        logger.okay("[qsr_machine] Daemon stopped")
        return True

    def status(self) -> None:
        pid = self.get_pid()
        if pid is None:
            logger.mesg("[qsr_machine] Status: not running")
            return
        logger.okay(f"[qsr_machine] Status: running (PID {pid})")
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
            logger.mesg("[qsr_machine] No log file found")
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
            logger.mesg("[qsr_machine] Log file is empty")


CLI_EPILOG = """
Examples:
  qsr machine run
  qsr machine run -b
  qsr machine run --auto-start
  qsr machine run -e http://localhost:27980,http://localhost:27981
  qsr machine status
  qsr machine logs -f
  qsr machine discover
  qsr machine health
"""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Run and manage the QSR machine proxy"
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
        "--on-conflict",
        choices=["report", "replace"],
        default="report",
        help=(
            "How to handle an existing qsr machine listener or daemon: "
            "report the conflict or replace it"
        ),
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help=(
            "When no QSR backends are running, start a compose deployment in the "
            "background and wait for healthy instances"
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=BACKEND_STARTUP_TIMEOUT_SEC,
        help=(
            "Seconds to wait for auto-started backends to become healthy "
            f"(default: {BACKEND_STARTUP_TIMEOUT_SEC:g})"
        ),
    )
    parser.add_argument(
        "--startup-poll-interval",
        type=float,
        default=BACKEND_STARTUP_POLL_INTERVAL_SEC,
        help=(
            "Polling interval in seconds while waiting for healthy backends "
            f"(default: {BACKEND_STARTUP_POLL_INTERVAL_SEC:g})"
        ),
    )
    parser.add_argument(
        "--compose-model-name",
        type=str,
        default=None,
        help=(
            "Model name to use for auto-started compose backends "
            f"(default: {COMPOSE_MODEL_NAME})"
        ),
    )
    parser.add_argument(
        "--compose-port",
        type=int,
        default=None,
        help=(
            "Compose backend base port for auto-started QSR containers "
            f"(default: {COMPOSE_SERVER_PORT})"
        ),
    )
    parser.add_argument(
        "--compose-project-name",
        type=str,
        default=None,
        help="Compose project name for auto-started backends",
    )
    parser.add_argument(
        "--compose-gpus",
        type=str,
        default=None,
        help="GPU IDs for auto-started backends (default: all healthy GPUs)",
    )
    parser.add_argument(
        "--compose-gpu-layout",
        choices=[GPU_LAYOUT_UNIFORM],
        default=None,
        help="Named GPU layout preset for auto-started backends",
    )
    parser.add_argument(
        "--compose-gpu-configs",
        type=str,
        default=None,
        help='Per-GPU config for auto-started backends: "GPU[:MODEL],..."',
    )
    parser.add_argument(
        "--compose-enable-sleep-mode",
        action="store_true",
        help="Enable vLLM sleep-mode endpoints on auto-started compose backends",
    )
    parser.add_argument(
        "-f", "--follow", action="store_true", help="Follow logs in real time"
    )
    parser.add_argument("--tail", type=int, default=50, help="Log lines to show")


def _discover_instances_from_docker(name_pattern: Optional[str]) -> list[QSRInstance]:
    return QSRInstanceDiscovery.discover(name_pattern)


def _build_auto_start_composer(args: argparse.Namespace) -> QSRComposer:
    composer_kwargs = {}
    if getattr(args, "compose_model_name", None):
        composer_kwargs["model_name"] = args.compose_model_name
    if getattr(args, "compose_port", None) is not None:
        composer_kwargs["port"] = args.compose_port
    if getattr(args, "compose_project_name", None):
        composer_kwargs["project_name"] = args.compose_project_name
    if getattr(args, "compose_gpus", None):
        composer_kwargs["gpu_ids"] = args.compose_gpus
    if getattr(args, "compose_gpu_layout", None):
        composer_kwargs["gpu_layout"] = args.compose_gpu_layout
    if getattr(args, "compose_gpu_configs", None):
        composer_kwargs["gpu_configs"] = parse_gpu_configs(args.compose_gpu_configs)
    if getattr(args, "compose_enable_sleep_mode", False):
        composer_kwargs["enable_sleep_mode"] = True
    return QSRComposer(**composer_kwargs)


def discover_instances(args: argparse.Namespace) -> list[QSRInstance]:
    if args.endpoints:
        endpoints = [
            endpoint.strip()
            for endpoint in args.endpoints.split(",")
            if endpoint.strip()
        ]
        instances = QSRInstanceDiscovery.from_endpoints(endpoints)
        logger.okay(f"[qsr_machine] Using {len(instances)} manual endpoints")
        return instances

    instances = _discover_instances_from_docker(args.name_pattern)
    if getattr(args, "auto_start", False):
        sleeping_endpoints = [
            instance.endpoint
            for instance in instances
            if get_backend_sleep_state(instance.endpoint) is True
        ]
        if sleeping_endpoints:
            logger.note(
                f"[qsr_machine] Found {len(sleeping_endpoints)} sleeping QSR backend(s); requesting wake-up"
            )
            _wake_sleeping_backends(
                sleeping_endpoints,
                timeout_sec=getattr(
                    args, "startup_timeout", BACKEND_STARTUP_TIMEOUT_SEC
                ),
                poll_interval_sec=getattr(
                    args,
                    "startup_poll_interval",
                    SLEEP_WAKE_POLL_INTERVAL_SEC,
                ),
                label="[qsr_machine]",
            )
            instances = _discover_instances_from_docker(args.name_pattern)

    instances = ensure_backend_instances(
        instances,
        enabled=getattr(args, "auto_start", False),
        manual_endpoints=bool(getattr(args, "endpoints", None)),
        service_label="[qsr_machine]",
        compose_factory=lambda: _build_auto_start_composer(args),
        rediscover=lambda: _discover_instances_from_docker(args.name_pattern),
        timeout_sec=getattr(args, "startup_timeout", BACKEND_STARTUP_TIMEOUT_SEC),
        poll_interval_sec=getattr(
            args,
            "startup_poll_interval",
            BACKEND_STARTUP_POLL_INTERVAL_SEC,
        ),
        allow_partial=True,
        settle_sec=BACKEND_DISCOVERY_SETTLE_SEC,
    )
    logger.okay(f"[qsr_machine] Discovered {len(instances)} QSR instances")
    return instances


def log_instances(instances: list[QSRInstance], show_health: bool = False) -> None:
    if not instances:
        logger.warn("× No QSR instances found")
        return

    dash_len = 96
    logger.note("=" * dash_len)
    if show_health:
        logger.note(
            f"{'GPU':<6} {'CONTAINER':<30} {'ENDPOINT':<25} {'MODEL':<20} {'STATUS':<8}"
        )
    else:
        logger.note(f"{'GPU':<6} {'CONTAINER':<30} {'ENDPOINT':<25} {'MODEL':<20}")
    logger.note("-" * dash_len)

    for instance in instances:
        gpu_text = str(instance.gpu_id) if instance.gpu_id is not None else "?"
        model_text = instance.model_name or "?"
        if show_health:
            status = logstr.okay("healthy") if instance.healthy else logstr.erro("sick")
            logger.mesg(
                f"{gpu_text:<6} {instance.container_name:<30} {instance.endpoint:<25} {model_text:<20} {status:<8}"
            )
        else:
            logger.mesg(
                f"{gpu_text:<6} {instance.container_name:<30} {instance.endpoint:<25} {model_text:<20}"
            )

    logger.note("=" * dash_len)
    if show_health:
        healthy = sum(1 for instance in instances if instance.healthy)
        logger.mesg(f"[qsr_machine] Healthy: {healthy}/{len(instances)}")


async def check_health(instances: list[QSRInstance]) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for instance in instances:
            quiet_health = _apply_qsr_docker_health(instance)
            if quiet_health is not None:
                continue
            try:
                response = await client.get(instance.health_url)
                instance.healthy = response.status_code == 200
            except Exception:
                instance.healthy = False
    log_instances(instances, show_health=True)


def run_from_args(args: argparse.Namespace) -> None:
    if args.action is None:
        raise ValueError("Action is required")

    daemon = QSRMachineDaemon()

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
        if not handle_port_conflicts(
            args.port,
            policy="replace",
            label="[qsr_machine]",
        ):
            return
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
                "× No QSR instances found. Use -e to specify endpoints manually."
            )
            return

        if args.background:
            if daemon.is_running():
                if getattr(args, "on_conflict", "report") == "report":
                    logger.warn(
                        f"[qsr_machine] Daemon already running (PID {daemon.get_pid()}). Use 'restart' or '--on-conflict replace' to replace it."
                    )
                    return
                logger.warn("[qsr_machine] Replacing existing daemon before restart")
                if not daemon.stop():
                    return
            if not handle_port_conflicts(
                args.port,
                policy=getattr(args, "on_conflict", "report"),
                label="[qsr_machine]",
            ):
                return
            daemon.start_background(sys.argv)
            return

        if not handle_port_conflicts(
            args.port,
            policy=getattr(args, "on_conflict", "report"),
            label="[qsr_machine]",
        ):
            return

        is_daemon = os.environ.get("_QSR_MACHINE_DAEMON") == "1"
        if is_daemon:
            daemon.write_pid()

        server = QSRMachineServer(
            instances=instances,
            port=args.port,
            timeout=args.timeout,
            discover_instances_fn=(
                None
                if args.endpoints
                else lambda: _discover_instances_from_docker(args.name_pattern)
            ),
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
