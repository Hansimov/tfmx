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
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from pydantic import BaseModel, Field
from tclogger import logger, logstr
from typing import Literal, Optional, Union
from webu import setup_swagger_ui

from .compose import MACHINE_PORT, MAX_CONCURRENT_REQUESTS
from .compose import get_display_shortcut, get_model_shortcut, normalize_model_key
from .gpu_runtime import GPURuntimeStats, query_gpu_runtime_stats
from .router import InstanceDescriptor, QWNRouter, parse_model_spec


PORT = MACHINE_PORT
MAX_CONCURRENT = MAX_CONCURRENT_REQUESTS
QWN_CONTAINER_IMAGE_PATTERN = "vllm"
DEFAULT_NAME_PATTERN = r"qwn[-_]"
HEALTH_REFRESH_INTERVAL_SEC = 5


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
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
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


class InstanceInfo(BaseModel):
    name: str = Field(...)
    endpoint: str = Field(...)
    gpu_id: int | None = Field(None)
    healthy: bool = Field(...)
    model_name: str = Field(default="")
    quant_method: str = Field(default="")
    quant_level: str = Field(default="")
    model_label: str = Field(default="")
    gpu_utilization_pct: float | None = Field(None)
    gpu_memory_used_mib: float | None = Field(None)
    gpu_memory_total_mib: float | None = Field(None)
    routing_pressure: float | None = Field(None)


class MachineStats(BaseModel):
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    active_requests: int = 0
    requests_per_instance: dict[str, int] = Field(default_factory=dict)


class InfoResponse(BaseModel):
    port: int
    instances: list[InstanceInfo] = Field(default_factory=list)
    stats: MachineStats = Field(default_factory=MachineStats)
    available_models: list[str] = Field(default_factory=list)


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

    def to_info(self) -> InstanceInfo:
        return InstanceInfo(
            name=self.container_name,
            endpoint=self.endpoint,
            gpu_id=self.gpu_id,
            healthy=self.healthy,
            model_name=self.model_name,
            quant_method=self.quant_method,
            quant_level=self.quant_level,
            model_label=self.model_name,
            gpu_utilization_pct=round(self._gpu_utilization_pct, 2),
            gpu_memory_used_mib=round(self._gpu_memory_used_mib, 2),
            gpu_memory_total_mib=round(self._gpu_memory_total_mib, 2),
            routing_pressure=round(self.routing_pressure, 4),
        )


@dataclass
class QWNStatsData:
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    active_requests: int = 0
    requests_per_instance: dict[str, int] = field(default_factory=dict)

    def to_model(self) -> MachineStats:
        return MachineStats(
            total_requests=self.total_requests,
            total_tokens=self.total_tokens,
            total_errors=self.total_errors,
            active_requests=self.active_requests,
            requests_per_instance=self.requests_per_instance,
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
        self._rr_lock: Optional[asyncio.Lock] = None
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

    def _get_idle_instance(
        self, model: str = "", quant: str = ""
    ) -> Optional[QWNInstance]:
        if model or quant:
            descriptors = self.router.find_instances(model, quant)
            endpoints = {descriptor.endpoint for descriptor in descriptors}
            candidates = [
                instance
                for instance in self.instances
                if instance.healthy
                and instance.is_idle
                and instance.endpoint in endpoints
            ]
        else:
            candidates = [
                instance
                for instance in self.instances
                if instance.healthy and instance.is_idle
            ]

        if not candidates:
            return None
        candidates.sort(
            key=lambda instance: (
                -instance.available_slots,
                instance.routing_pressure,
                instance._latency_ms if instance._latency_ms > 0 else float("inf"),
                instance.gpu_id if instance.gpu_id is not None else 999,
            )
        )
        return candidates[0]

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
        app.post("/v1/chat/completions", response_model=ChatCompletionResponse)(
            self.chat_completions
        )
        app.post("/chat", response_model=ChatCompletionResponse)(self.chat_form)
        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        self._rr_lock = asyncio.Lock()

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

    async def _check_instance_health(self, instance: QWNInstance) -> bool:
        try:
            started_at = time.perf_counter()
            response = await self._client.get(instance.health_url)
            instance.healthy = response.status_code == 200
            instance._latency_ms = (
                (time.perf_counter() - started_at) * 1000 if instance.healthy else 0.0
            )
            return instance.healthy
        except Exception:
            instance.healthy = False
            instance._latency_ms = 0.0
            return False

    async def _discover_instance_models(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))

        for instance in self.instances:
            if not instance.healthy or instance.model_name:
                continue
            try:
                response = await self._client.get(instance.models_url)
                if response.status_code != 200:
                    continue
                payload = response.json()
                models = payload.get("data", [])
                if not models:
                    continue
                model_id = models[0].get("id", "")
                instance.model_name = model_id
                base_model, quant_level = parse_model_spec(model_id)
                if quant_level:
                    instance.quant_method = "awq"
                    instance.quant_level = quant_level
                elif "awq" in model_id.lower():
                    instance.quant_method = "awq"
                logger.mesg(
                    f"[qwn_machine] {instance.container_name}: model={model_id}"
                )
            except Exception:
                pass

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
        return InfoResponse(
            port=self.port,
            instances=[instance.to_info() for instance in self.instances],
            stats=self.stats.to_model(),
            available_models=self.router.get_available_models(),
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
        max_tokens: int = Form(default=512, description="Maximum tokens"),
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
        self, model_field: str = ""
    ) -> tuple[QWNInstance, str, str]:
        requested_model = ""
        requested_quant = ""
        if model_field:
            requested_model, requested_quant = parse_model_spec(model_field)

        instance = self._get_idle_instance(model=requested_model, quant=requested_quant)
        if instance is None:
            for _ in range(10):
                await asyncio.sleep(0.1)
                instance = self._get_idle_instance(
                    model=requested_model, quant=requested_quant
                )
                if instance is not None:
                    break

        if instance is None:
            instance = self._get_idle_instance()

        if instance is None:
            self.stats.total_errors += 1
            raise HTTPException(status_code=503, detail="No available QWN instances")

        return instance, requested_model, requested_quant

    def _rewrite_model_in_body(self, body: bytes, instance: QWNInstance) -> bytes:
        if not instance.model_name:
            return body
        try:
            payload = orjson.loads(body)
            payload["model"] = instance.model_name
            return orjson.dumps(payload)
        except Exception:
            return body

    async def _forward_stream(
        self, body: bytes, model_field: str = ""
    ) -> StreamingResponse:
        self.stats.total_requests += 1
        self.stats.active_requests += 1
        try:
            instance, _, _ = await self._acquire_instance(model_field)
        except HTTPException:
            self.stats.active_requests = max(0, self.stats.active_requests - 1)
            raise

        instance._active_requests += 1
        model_label = self._get_model_label(instance)
        self.stats.requests_per_instance[instance.container_name] = (
            self.stats.requests_per_instance.get(instance.container_name, 0) + 1
        )
        body = self._rewrite_model_in_body(body, instance)

        async def stream_generator():
            try:
                async with self._client.stream(
                    "POST",
                    instance.chat_url,
                    content=body,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status_code != 200:
                        error_text = (await response.aread()).decode(
                            "utf-8", errors="replace"
                        )
                        yield f"data: {orjson.dumps({'error': {'message': error_text, 'type': 'upstream_error', 'code': response.status_code}}).decode()}\n\n"
                        yield "data: [DONE]\n\n"
                        self.stats.total_errors += 1
                        return

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            payload = line[6:].strip()
                            if payload == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            try:
                                chunk = orjson.loads(payload)
                                if model_label:
                                    chunk["model"] = model_label
                                usage = chunk.get("usage")
                                if usage:
                                    self.stats.total_tokens += usage.get(
                                        "total_tokens", 0
                                    )
                                yield f"data: {orjson.dumps(chunk).decode()}\n\n"
                            except Exception:
                                yield f"data: {payload}\n\n"
                        elif line.startswith(":"):
                            yield f"{line}\n"
            except Exception as exc:
                self.stats.total_errors += 1
                logger.warn(f"× Stream error: {exc}")
                error_payload = {
                    "error": {
                        "message": str(exc),
                        "type": "proxy_error",
                        "code": 500,
                    }
                }
                yield f"data: {orjson.dumps(error_payload).decode()}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                instance._active_requests = max(0, instance._active_requests - 1)
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
            instance, _, _ = await self._acquire_instance(model_field)
            instance._active_requests += 1
            body = self._rewrite_model_in_body(body, instance)
            response = await self._client.post(
                instance.chat_url,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            self.stats.requests_per_instance[instance.container_name] = (
                self.stats.requests_per_instance.get(instance.container_name, 0) + 1
            )

            if response.status_code != 200:
                self.stats.total_errors += 1
                raise HTTPException(
                    status_code=response.status_code, detail=response.text
                )

            payload = orjson.loads(response.content)
            usage = payload.get("usage", {})
            self.stats.total_tokens += usage.get("total_tokens", 0)
            return ChatCompletionResponse(
                id=payload.get("id", ""),
                object=payload.get("object", "chat.completion"),
                created=payload.get("created", 0),
                model=self._get_model_label(instance) or payload.get("model", ""),
                choices=[
                    ChatChoiceDelta(**choice) for choice in payload.get("choices", [])
                ],
                usage=UsageInfo(**usage),
            )
        except HTTPException:
            raise
        except Exception as exc:
            self.stats.total_errors += 1
            logger.warn(f"× Chat completion error: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            self.stats.active_requests = max(0, self.stats.active_requests - 1)
            if "instance" in locals():
                instance._active_requests = max(0, instance._active_requests - 1)

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
