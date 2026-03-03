"""QVL (Qwen3-VL) Machine Manager

A load-balanced proxy server that distributes chat completion requests
across multiple vLLM Docker instances running on different GPUs.
"""

# ANCHOR[id=qvl-machine-clis]
CLI_EPILOG = """
Examples:
  # Start machine server (foreground, auto-discover vLLM containers)
  qvl_machine run                   # Start on default port 29800
  qvl_machine run -p 29800          # Start on specific port

  # Start as background daemon
  qvl_machine run -b                # Run in background (logs to file)
  qvl_machine run -b -e "http://localhost:29880,http://localhost:29881"

  # Filter containers by name pattern
  qvl_machine run -n "qvl--qwen"    # Only match containers with this pattern

  # Manual endpoint specification (skip auto-discovery)
  qvl_machine run -e "http://localhost:29880,http://localhost:29881"

  # Performance tracking
  qvl_machine run --perf-track      # Enable detailed performance tracking

  # Service management (for background daemon)
  qvl_machine stop                  # Stop the background service
  qvl_machine restart               # Restart (stop + run -b)
  qvl_machine restart -e "..."      # Restart with new endpoints
  qvl_machine status                # Check if service is running
  qvl_machine logs                  # View recent logs (last 50 lines)
  qvl_machine logs -f               # Follow logs in real-time
  qvl_machine logs --tail 200       # View last 200 lines

  # Check discovered instances without starting server
  qvl_machine discover              # List all discovered vLLM instances

  # Health check all instances
  qvl_machine health                # Check health of all instances
"""

import argparse
import asyncio
import base64
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
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pathlib import Path
from pydantic import BaseModel, Field
from tclogger import logger, logstr
from typing import Annotated, Literal, Optional, Union
from webu import setup_swagger_ui

from .compose import MAX_CONCURRENT_REQUESTS, MACHINE_PORT, SERVER_PORT
from .compose import get_model_shortcut, get_display_shortcut, normalize_model_key
from .router import QVLRouter, InstanceDescriptor, parse_model_spec


PORT = MACHINE_PORT
MAX_CONCURRENT = MAX_CONCURRENT_REQUESTS
VLLM_CONTAINER_IMAGE_PATTERN = "vllm"


# ANCHOR[id=qvl-machine-models]


class TextContent(BaseModel):
    """Text content part."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")


class ImageURL(BaseModel):
    """Image URL reference."""

    url: str = Field(
        ...,
        description="Image URL or base64 data URI (data:image/jpeg;base64,...)",
    )
    detail: str = Field(default="auto", description="Detail level: auto, low, high")


class ImageContent(BaseModel):
    """Image content part."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class VideoURL(BaseModel):
    """Video URL reference."""

    url: str = Field(..., description="Video URL or base64 data URI")


class VideoContent(BaseModel):
    """Video content part."""

    type: Literal["video_url"] = "video_url"
    video_url: VideoURL


ContentPart = Union[TextContent, ImageContent, VideoContent]


class ChatMessage(BaseModel):
    """A chat message with role and content.

    Content can be a simple string or a list of multimodal content parts.
    """

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Message role"
    )
    content: Union[str, list[ContentPart]] = Field(
        ...,
        description=(
            "Message content: plain text string, or a list of content parts "
            "(text, image_url, video_url) for multimodal inputs"
        ),
        examples=[
            "Hello, what can you do?",
            [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,..."},
                },
            ],
        ],
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with multimodal support.

    Supports text, images (URL or base64), and video content.
    """

    model: str = Field(
        default="",
        description=(
            "Model name or shortcut (e.g., '4b-thinking', '8b-instruct:4bit'). "
            "Leave empty for default model."
        ),
    )
    messages: list[ChatMessage] = Field(
        ...,
        description="Chat messages",
        examples=[
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ]
        ],
    )
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Enable streaming")


class ChatChoiceDelta(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: dict = Field(default_factory=dict)
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default="", description="Completion ID")
    object: str = Field(default="chat.completion")
    created: int = Field(default=0, description="Unix timestamp")
    model: str = Field(default="", description="Model used")
    choices: list[ChatChoiceDelta] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str = Field(..., description="Health status", examples=["healthy"])
    healthy: int = Field(..., description="Number of healthy instances")
    total: int = Field(..., description="Total number of instances")


class InstanceInfo(BaseModel):
    """Information about a single vLLM instance."""

    name: str = Field(..., description="Container name")
    endpoint: str = Field(..., description="HTTP endpoint URL")
    gpu_id: Optional[int] = Field(None, description="GPU device ID")
    healthy: bool = Field(..., description="Whether instance is healthy")
    model_name: str = Field(
        "", description="Model name (e.g., Qwen/Qwen3-VL-8B-Instruct)"
    )
    quant_method: str = Field(
        "", description="Quantization method (awq, bitsandbytes, etc.)"
    )
    quant_level: str = Field("", description="Quantization level (4bit)")
    model_label: str = Field(
        "", description="Short model label (e.g., 8B-Instruct:4bit)"
    )


class MachineStats(BaseModel):
    """Statistics for the machine."""

    total_requests: int = Field(0, description="Total requests processed")
    total_tokens: int = Field(0, description="Total tokens generated")
    total_errors: int = Field(0, description="Total errors")
    active_requests: int = Field(0, description="Currently active requests")
    requests_per_instance: dict[str, int] = Field(
        default_factory=dict, description="Request count per instance"
    )


class InfoResponse(BaseModel):
    """Response model for info endpoint."""

    port: int = Field(..., description="Machine server port")
    instances: list[InstanceInfo] = Field(..., description="vLLM instances")
    stats: MachineStats = Field(..., description="Machine statistics")
    available_models: list[str] = Field(
        default_factory=list, description="Available model labels"
    )


@dataclass
class VLLMInstance:
    """Represents a single vLLM Docker instance."""

    container_name: str
    host: str
    port: int
    gpu_id: Optional[int] = None
    healthy: bool = False
    _active_requests: int = 0
    model_name: str = ""
    quant_method: str = ""
    quant_level: str = ""

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
        gpu_info = f"GPU{self.gpu_id}" if self.gpu_id is not None else "GPU?"
        model_info = ""
        if self.model_name:
            shortcut = get_model_shortcut(self.model_name)
            quant = f":{self.quant_level}" if self.quant_level else ""
            model_info = f" [{shortcut}{quant}]"
        return (
            f"VLLMInstance({status} {self.container_name} "
            f"@ {self.endpoint}, {gpu_info}{model_info})"
        )

    def to_info(self) -> InstanceInfo:
        shortcut = get_model_shortcut(self.model_name)
        quant = f":{self.quant_level}" if self.quant_level else ""
        label = f"{shortcut}{quant}" if shortcut else ""
        return InstanceInfo(
            name=self.container_name,
            endpoint=self.endpoint,
            gpu_id=self.gpu_id,
            healthy=self.healthy,
            model_name=self.model_name,
            quant_method=self.quant_method,
            quant_level=self.quant_level,
            model_label=label,
        )


@dataclass
class VLLMStatsData:
    """Internal statistics tracking."""

    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    active_requests: int = 0
    requests_per_instance: dict = field(default_factory=dict)

    def to_model(self) -> MachineStats:
        return MachineStats(
            total_requests=self.total_requests,
            total_tokens=self.total_tokens,
            total_errors=self.total_errors,
            active_requests=self.active_requests,
            requests_per_instance=self.requests_per_instance,
        )


class VLLMInstanceDiscovery:
    """Discovers running vLLM Docker instances."""

    @staticmethod
    def discover(name_pattern: Optional[str] = None) -> list[VLLMInstance]:
        """Discover running vLLM containers and their exposed ports.

        Args:
            name_pattern: Optional regex pattern to filter container names

        Returns:
            List of discovered VLLMInstance objects
        """
        try:
            if name_pattern:
                cmd = (
                    f"docker ps --format '{{{{.Names}}}}|{{{{.Image}}}}|{{{{.Ports}}}}' "
                    f"--filter 'name={name_pattern}'"
                )
            else:
                cmd = "docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}'"

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warn(f"× Docker command failed: {result.stderr}")
                return []

            if not result.stdout.strip():
                logger.note("[qvl_machine] No running containers found")
                return []

            instances = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("|")
                if len(parts) < 3:
                    continue

                container_name, image, ports = parts[0], parts[1], parts[2]

                # Filter by vLLM image pattern
                if VLLM_CONTAINER_IMAGE_PATTERN not in image:
                    continue

                # Additional name pattern filter if specified
                if name_pattern and not re.search(name_pattern, container_name):
                    continue

                host_port = VLLMInstanceDiscovery._extract_host_port(
                    ports, container_name
                )
                if host_port is None:
                    continue

                gpu_id = VLLMInstanceDiscovery._extract_gpu_id(container_name)

                instance = VLLMInstance(
                    container_name=container_name,
                    host="localhost",
                    port=host_port,
                    gpu_id=gpu_id,
                )
                instances.append(instance)

            # Sort by GPU ID
            instances.sort(key=lambda x: (x.gpu_id if x.gpu_id is not None else 999))
            return instances

        except Exception as e:
            logger.warn(f"× Failed to discover vLLM instances: {e}")
            return []

    @staticmethod
    def _extract_host_port(ports_str: str, container_name: str = "") -> Optional[int]:
        """Extract host port from Docker port mapping or container inspect."""
        # Bridge mode: "0.0.0.0:29880->8000/tcp"
        match = re.search(r"(?:0\.0\.0\.0|::):(\d+)->", ports_str)
        if match:
            return int(match.group(1))

        # Host network mode: get --port from container args
        if not ports_str and container_name:
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        container_name,
                        "--format",
                        "{{.Args}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    args_str = result.stdout.strip()
                    port_match = re.search(r"--port\s+(\d+)", args_str)
                    if port_match:
                        return int(port_match.group(1))
            except Exception:
                pass

        return None

    @staticmethod
    def _extract_gpu_id(container_name: str) -> Optional[int]:
        """Extract GPU ID from container name (e.g., 'qvl--xxx--gpu0' -> 0)."""
        match = re.search(r"--gpu(\d+)", container_name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def from_endpoints(endpoints: list[str]) -> list[VLLMInstance]:
        """Create instances from manual endpoint specifications."""
        instances = []
        for i, endpoint in enumerate(endpoints):
            endpoint = endpoint.strip()
            if not endpoint:
                continue

            match = re.match(r"https?://([^:]+):(\d+)", endpoint)
            if match:
                host, port = match.group(1), int(match.group(2))
            else:
                try:
                    port = int(endpoint)
                    host = "localhost"
                except ValueError:
                    continue

            instance = VLLMInstance(
                container_name=f"manual-{i}",
                host=host,
                port=port,
                gpu_id=i,
            )
            instances.append(instance)

        return instances


class QVLMachineServer:
    """FastAPI server that proxies chat requests to multiple vLLM instances.

    Routes:
        POST /v1/chat/completions - Chat completion (distributed)
        GET  /health              - Health check
        GET  /info                - Instance info and stats
    """

    def __init__(
        self,
        instances: list[VLLMInstance],
        port: int = PORT,
        timeout: float = 120.0,
        enable_perf_tracking: bool = False,
    ):
        self.instances = instances
        self.port = port
        self.timeout = timeout
        self.stats = VLLMStatsData()
        self._client: Optional[httpx.AsyncClient] = None
        self._health_task: Optional[asyncio.Task] = None
        self.enable_perf_tracking = enable_perf_tracking

        # Router for model/quant-aware request routing
        self.router = QVLRouter()

        # Round-robin index for distributing requests
        self._rr_lock: Optional[asyncio.Lock] = None

        self.app = self._create_app()

    def _build_router(self) -> None:
        """Build router from discovered instances."""
        self.router = QVLRouter()
        for inst in self.instances:
            desc = InstanceDescriptor(
                model_name=inst.model_name,
                quant_method=inst.quant_method,
                quant_level=inst.quant_level,
                endpoint=inst.endpoint,
                gpu_id=inst.gpu_id,
                instance_id=inst.container_name,
                healthy=inst.healthy,
            )
            self.router.register(desc)

    def get_healthy_instances(self) -> list[VLLMInstance]:
        return [i for i in self.instances if i.healthy]

    def _get_idle_instance(
        self, model: str = "", quant: str = ""
    ) -> Optional[VLLMInstance]:
        """Get idle instance, optionally filtered by model/quant."""
        if model or quant:
            # Use router to find matching instances
            matching_descs = self.router.find_instances(model, quant)
            matching_endpoints = {d.endpoint for d in matching_descs}
            candidates = [
                i
                for i in self.instances
                if i.healthy and i.is_idle and i.endpoint in matching_endpoints
            ]
        else:
            candidates = [i for i in self.instances if i.healthy and i.is_idle]

        if not candidates:
            return None
        candidates.sort(key=lambda i: i.available_slots, reverse=True)
        return candidates[0]

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="QVL Machine",
            description=(
                "Load-balanced proxy for Qwen3-VL vLLM instances.\n\n"
                "## Endpoints\n\n"
                "- **POST /v1/chat/completions** — OpenAI-compatible chat API "
                "(supports multimodal: text, images, video)\n"
                "- **POST /chat** — Simplified form-based chat with file uploads\n"
                "- **GET /health** — Health check\n"
                "- **GET /info** — Instance info and stats\n"
            ),
            version="1.0.0",
            lifespan=self._lifespan,
            docs_url=None,
            redoc_url=None,
        )

        setup_swagger_ui(app)

        app.get(
            "/health",
            response_model=HealthResponse,
            summary="Health check",
        )(self.health)

        app.get(
            "/info",
            response_model=InfoResponse,
            summary="Machine info",
        )(self.info)

        app.post(
            "/v1/chat/completions",
            response_model=ChatCompletionResponse,
            summary="Chat completion (OpenAI-compatible)",
            description=(
                "Forward chat completion request to a vLLM instance. "
                "Supports multimodal messages with text, images (base64 or URL), "
                "and video content.\n\n"
                "**Model routing**: Use short names like `4b-thinking`, "
                "`8b-instruct:4bit`, or leave empty for the default model."
            ),
        )(self.chat_completions)

        app.post(
            "/chat",
            response_model=ChatCompletionResponse,
            summary="Chat (form-based, with file uploads)",
            description=(
                "Simplified chat endpoint for the Swagger UI. "
                "Upload images/PDFs/videos as files and type your message — "
                "no need to manually construct JSON message arrays.\n\n"
                "Each uploaded file is automatically converted to the appropriate "
                "multimodal content type based on its MIME type."
            ),
        )(self.chat_form)

        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        self._rr_lock = asyncio.Lock()

        await self.health_check_all()
        await self._discover_instance_models()
        self._build_router()

        healthy = self.get_healthy_instances()
        if not healthy:
            logger.warn("× No healthy vLLM instances available at startup")

        self._health_task = asyncio.create_task(self._periodic_health_check())

        logger.okay(f"[qvl_machine] Started on port {self.port}")
        healthy_str = logstr.okay(len(healthy))
        total_str = logstr.mesg(len(self.instances))
        logger.mesg(f"[qvl_machine] Healthy instances: {healthy_str}/{total_str}")
        if self.router:
            models = self.router.get_available_models()
            if models:
                logger.mesg(f"[qvl_machine] Models: {models}")

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
            await asyncio.sleep(30)
            await self.health_check_all()

    async def health_check_all(self) -> None:
        if not self._client:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))

        tasks = [self._check_instance_health(inst) for inst in self.instances]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Update router health status
        for inst in self.instances:
            for desc in self.router.instances:
                if desc.endpoint == inst.endpoint:
                    desc.healthy = inst.healthy

    async def _check_instance_health(self, instance: VLLMInstance) -> bool:
        try:
            resp = await self._client.get(instance.health_url)
            instance.healthy = resp.status_code == 200
            return instance.healthy
        except Exception:
            instance.healthy = False
            return False

    async def health(self) -> HealthResponse:
        healthy = self.get_healthy_instances()
        response = HealthResponse(
            status="healthy" if healthy else "unhealthy",
            healthy=len(healthy),
            total=len(self.instances),
        )
        if not healthy:
            raise HTTPException(status_code=503, detail=response.model_dump())
        return response

    async def _discover_instance_models(self) -> None:
        """Query each instance's /v1/models to discover model names."""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))

        for inst in self.instances:
            if not inst.healthy or inst.model_name:
                continue
            try:
                resp = await self._client.get(f"{inst.endpoint}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models_data = data.get("data", [])
                    if models_data:
                        model_id = models_data[0].get("id", "")
                        inst.model_name = model_id
                        # Try to infer quant from model name
                        if "AWQ" in model_id.upper():
                            inst.quant_method = "awq"
                        logger.mesg(
                            f"[qvl_machine] {inst.container_name}: model={model_id}"
                        )
            except Exception:
                pass

    async def info(self) -> InfoResponse:
        return InfoResponse(
            port=self.port,
            instances=[inst.to_info() for inst in self.instances],
            stats=self.stats.to_model(),
            available_models=self.router.get_available_models(),
        )

    async def chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Forward chat completion request to an idle vLLM instance.

        Routes based on model/quant in request body if specified.
        Uses least-loaded instance for unspecified requests.
        """
        body = orjson.dumps(request.model_dump(exclude_none=True))
        return await self._forward_chat(body, request.model)

    async def chat_form(
        self,
        text: Annotated[
            str,
            Form(description="Your message text"),
        ],
        files: Annotated[
            list[UploadFile],
            File(description="Upload images, PDFs, or videos"),
        ] = [],
        system_prompt: Annotated[
            str,
            Form(description="Optional system prompt"),
        ] = "",
        model: Annotated[
            str,
            Form(
                description=(
                    "Model shortcut (e.g., '4b-thinking', '8b-instruct:4bit'). "
                    "Leave empty for default."
                )
            ),
        ] = "",
        max_tokens: Annotated[
            int, Form(description="Maximum tokens to generate")
        ] = 512,
        temperature: Annotated[
            float, Form(description="Sampling temperature (0.0-2.0)")
        ] = 0.7,
    ) -> ChatCompletionResponse:
        """Chat with file uploads — no JSON needed.

        Upload images/PDFs/videos and type your message.
        Files are auto-converted to multimodal content parts.
        """
        # Build content parts from text + uploaded files
        content_parts: list[dict] = []

        for file in files:
            file_data = await file.read()
            b64 = base64.b64encode(file_data).decode("utf-8")
            mime = file.content_type or "application/octet-stream"

            if mime.startswith("image/"):
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
            elif mime.startswith("video/"):
                content_parts.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
            else:
                # PDF and other files: send as image (vLLM can handle some)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )

        content_parts.append({"type": "text", "text": text})

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Use content_parts list if multimodal, plain string if text-only
        if files:
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": text})

        req_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body = orjson.dumps(req_data)
        return await self._forward_chat(body, model)

    async def _forward_chat(
        self, body: bytes, model_field: str = ""
    ) -> ChatCompletionResponse:
        """Forward a chat request body to a vLLM instance and return typed response."""
        t0 = time.perf_counter()
        self.stats.total_requests += 1
        self.stats.active_requests += 1

        try:
            # Parse model/quant for routing
            req_model = ""
            req_quant = ""
            if model_field:
                req_model, req_quant = parse_model_spec(model_field)

            # Get an idle instance (with optional model/quant filter)
            instance = self._get_idle_instance(model=req_model, quant=req_quant)
            if instance is None:
                # Wait briefly for a slot to free up
                for _ in range(10):
                    await asyncio.sleep(0.1)
                    instance = self._get_idle_instance(model=req_model, quant=req_quant)
                    if instance:
                        break

            if instance is None:
                # Fall back to any idle instance
                instance = self._get_idle_instance()

            if instance is None:
                self.stats.total_errors += 1
                raise HTTPException(
                    status_code=503, detail="No available vLLM instances"
                )

            instance._active_requests += 1

            try:
                # Rewrite model name to match vLLM's actual model name
                # (proxy accepts short names like "2b-instruct:4bit" but vLLM
                #  needs the full HF repo name from its /v1/models endpoint)
                if instance.model_name:
                    try:
                        req_data = orjson.loads(body)
                        req_data["model"] = instance.model_name
                        body = orjson.dumps(req_data)
                    except Exception:
                        pass

                # Forward to vLLM
                resp = await self._client.post(
                    instance.chat_url,
                    content=body,
                    headers={"Content-Type": "application/json"},
                )

                # Track stats
                instance_name = instance.container_name
                self.stats.requests_per_instance[instance_name] = (
                    self.stats.requests_per_instance.get(instance_name, 0) + 1
                )

                # Track tokens from response
                resp_data = None
                if resp.status_code == 200:
                    try:
                        resp_data = orjson.loads(resp.content)
                        usage = resp_data.get("usage", {})
                        self.stats.total_tokens += usage.get("total_tokens", 0)
                    except Exception:
                        pass

                latency = time.perf_counter() - t0
                if self.enable_perf_tracking:
                    logger.mesg(
                        f"[qvl_machine] {instance_name} "
                        f"status={resp.status_code} latency={latency:.2f}s"
                    )

                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=resp.text,
                    )

                # Parse vLLM response into typed model
                if resp_data is None:
                    resp_data = orjson.loads(resp.content)

                return ChatCompletionResponse(
                    id=resp_data.get("id", ""),
                    object=resp_data.get("object", "chat.completion"),
                    created=resp_data.get("created", 0),
                    model=resp_data.get("model", ""),
                    choices=[
                        ChatChoiceDelta(**c) for c in resp_data.get("choices", [])
                    ],
                    usage=UsageInfo(**resp_data.get("usage", {})),
                )

            finally:
                instance._active_requests = max(0, instance._active_requests - 1)

        except HTTPException:
            raise
        except Exception as e:
            self.stats.total_errors += 1
            logger.warn(f"× Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.stats.active_requests = max(0, self.stats.active_requests - 1)

    def run(self) -> None:
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info",
        )

    async def run_async(self) -> None:
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()


# ── Daemon Management ────────────────────────────────────────────────

# Default paths for PID and log files
_CACHE_DIR = Path.home() / ".cache" / "tfmx"
_PID_FILE = _CACHE_DIR / "qvl_machine.pid"
_LOG_FILE = _CACHE_DIR / "qvl_machine.log"


class QVLMachineDaemon:
    """Manages the qvl_machine process lifecycle (start, stop, status, logs).

    In background mode (``-b``), the server forks into a daemon process with
    stdout/stderr redirected to a log file. A PID file tracks the process.

    File locations:
        - PID: ``~/.cache/tfmx/qvl_machine.pid``
        - Log: ``~/.cache/tfmx/qvl_machine.log``
    """

    def __init__(
        self,
        pid_file: Path = _PID_FILE,
        log_file: Path = _LOG_FILE,
    ):
        self.pid_file = pid_file
        self.log_file = log_file
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)

    def get_pid(self) -> Optional[int]:
        """Read PID from file. Returns None if not found or stale."""
        if not self.pid_file.exists():
            return None
        try:
            pid = int(self.pid_file.read_text().strip())
            # Verify process is actually alive
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            # PID file is stale - clean it up
            self.pid_file.unlink(missing_ok=True)
            return None

    def is_running(self) -> bool:
        """Check if the daemon process is alive."""
        return self.get_pid() is not None

    def write_pid(self) -> None:
        """Write current process PID to file."""
        self.pid_file.write_text(str(os.getpid()))

    def remove_pid(self) -> None:
        """Remove PID file."""
        self.pid_file.unlink(missing_ok=True)

    def start_background(self, argv: list[str]) -> None:
        """Fork the current command into a background daemon process.

        Rebuilds the command line from sys.argv with ``-b`` removed,
        then launches it as a detached subprocess with output to log file.
        """
        # Build command: same as current invocation but without -b/--background
        cmd = [sys.executable, "-m", "tfmx.qvls.machine"]
        for arg in argv[1:]:
            if arg in ("-b", "--background"):
                continue
            cmd.append(arg)

        self._ensure_cache_dir()
        log_fh = open(self.log_file, "a")

        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from terminal
            env={**os.environ, "_QVL_MACHINE_DAEMON": "1"},
        )

        # Write PID of the child
        self.pid_file.write_text(str(proc.pid))

        logger.okay(f"[qvl_machine] Started in background (PID {proc.pid})")
        logger.mesg(f"  Log: {self.log_file}")
        logger.mesg(f"  PID: {self.pid_file}")

    def stop(self) -> bool:
        """Stop the background daemon. Returns True if a process was stopped."""
        pid = self.get_pid()
        if pid is None:
            logger.mesg("[qvl_machine] No running daemon found")
            return False

        logger.mesg(f"[qvl_machine] Stopping daemon (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit (up to 10s)
            for _ in range(100):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except ProcessLookupError:
                    break
            else:
                # Force kill if still alive after 10s
                logger.warn(f"[qvl_machine] Force killing PID {pid}")
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warn(f"x Permission denied killing PID {pid}")
            return False

        self.remove_pid()
        logger.okay("[qvl_machine] Daemon stopped")
        return True

    def status(self) -> None:
        """Print daemon status information."""
        pid = self.get_pid()
        if pid is None:
            logger.mesg("[qvl_machine] Status: not running")
            return

        logger.okay(f"[qvl_machine] Status: running (PID {pid})")
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
        """Show daemon log output.

        Args:
            follow: If True, use ``tail -f`` to follow logs in real-time.
            tail: Number of lines to show from the end of the log file.
        """
        if not self.log_file.exists():
            logger.mesg("[qvl_machine] No log file found")
            return

        if follow:
            try:
                subprocess.run(
                    ["tail", "-f", "-n", str(tail), str(self.log_file)],
                )
            except KeyboardInterrupt:
                pass
        else:
            try:
                result = subprocess.run(
                    ["tail", "-n", str(tail), str(self.log_file)],
                    capture_output=True,
                    text=True,
                )
                if result.stdout:
                    print(result.stdout, end="")
                else:
                    logger.mesg("[qvl_machine] Log file is empty")
            except Exception as e:
                logger.warn(f"x Failed to read logs: {e}")


class QVLMachineArgParser:
    """Argument parser for QVL Machine CLI."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="QVL Machine - Load-balanced proxy for vLLM instances",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        self.parser.add_argument(
            "action",
            nargs="?",
            choices=["run", "discover", "health", "stop", "restart", "status", "logs"],
            default=None,
            help=(
                "Action: run (start server), stop (stop daemon), "
                "restart (restart daemon), status (check daemon), "
                "logs (view logs), discover (list instances), "
                "health (check health)"
            ),
        )
        self.parser.add_argument(
            "-p",
            "--port",
            type=int,
            default=PORT,
            help=f"Machine server port (default: {PORT})",
        )
        self.parser.add_argument(
            "-n",
            "--name-pattern",
            type=str,
            default=None,
            help="Regex pattern to filter container names",
        )
        self.parser.add_argument(
            "-e",
            "--endpoints",
            type=str,
            default=None,
            help="Comma-separated list of vLLM endpoints (skip auto-discovery)",
        )
        self.parser.add_argument(
            "-t",
            "--timeout",
            type=float,
            default=120.0,
            help="Request timeout in seconds (default: 120)",
        )
        self.parser.add_argument(
            "--perf-track",
            action="store_true",
            help="Enable detailed performance tracking",
        )
        self.parser.add_argument(
            "-b",
            "--background",
            action="store_true",
            help="Run as background daemon (logs to file instead of terminal)",
        )
        self.parser.add_argument(
            "-f",
            "--follow",
            action="store_true",
            help="Follow logs in real-time (for 'logs' action)",
        )
        self.parser.add_argument(
            "--tail",
            type=int,
            default=50,
            help="Number of log lines to show (default: 50)",
        )


def discover_instances(args) -> list[VLLMInstance]:
    """Discover or create vLLM instances based on args."""
    if args.endpoints:
        endpoints = [e.strip() for e in args.endpoints.split(",")]
        instances = VLLMInstanceDiscovery.from_endpoints(endpoints)
        logger.okay(f"[qvl_machine] Using {len(instances)} manual endpoints")
    else:
        instances = VLLMInstanceDiscovery.discover(args.name_pattern)
        logger.okay(f"[qvl_machine] Discovered {len(instances)} vLLM instances")
    return instances


def log_instances(instances: list[VLLMInstance], show_health: bool = False) -> None:
    """Print discovered instances."""
    if not instances:
        logger.warn("× No vLLM instances found")
        return

    dash_len = 100
    logger.note("=" * dash_len)

    if show_health:
        logger.note(
            f"{'GPU':<6} {'CONTAINER':<35} {'ENDPOINT':<25} {'MODEL':<22} {'STATUS':<8}"
        )
    else:
        logger.note(f"{'GPU':<6} {'CONTAINER':<35} {'ENDPOINT':<25} {'MODEL':<22}")

    logger.note("-" * dash_len)

    for inst in instances:
        gpu_str = str(inst.gpu_id) if inst.gpu_id is not None else "?"
        model_info = inst.to_info().model_label or "?"
        if show_health:
            if inst.healthy:
                status = logstr.okay("✓ healthy")
            else:
                status = logstr.erro("× sick")
            logger.mesg(
                f"{gpu_str:<6} {inst.container_name:<35} "
                f"{inst.endpoint:<25} {model_info:<22} {status:<8}"
            )
        else:
            logger.mesg(
                f"{gpu_str:<6} {inst.container_name:<35} "
                f"{inst.endpoint:<25} {model_info:<22}"
            )

    logger.note("=" * dash_len)

    if show_health:
        healthy = sum(1 for i in instances if i.healthy)
        logger.mesg(f"[qvl_machine] Healthy: {healthy}/{len(instances)}")


async def check_health(instances: list[VLLMInstance]) -> None:
    """Check health of all instances and print status."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for inst in instances:
            try:
                resp = await client.get(inst.health_url)
                inst.healthy = resp.status_code == 200
            except Exception:
                inst.healthy = False

    log_instances(instances, show_health=True)


def main():
    """Main entry point for qvl_machine CLI.

    Supports foreground and background (daemon) modes:
    - ``qvl_machine run``       — foreground server
    - ``qvl_machine run -b``    — background daemon
    - ``qvl_machine stop``      — stop daemon
    - ``qvl_machine restart``   — restart daemon
    - ``qvl_machine status``    — check if daemon is running
    - ``qvl_machine logs [-f]`` — view daemon logs
    """
    arg_parser = QVLMachineArgParser()
    args = arg_parser.args

    if args.action is None:
        arg_parser.parser.print_help()
        return

    daemon = QVLMachineDaemon()

    # ── Service management actions (no instance discovery needed) ──

    if args.action == "stop":
        daemon.stop()
        return

    if args.action == "status":
        daemon.status()
        return

    if args.action == "logs":
        daemon.show_logs(
            follow=getattr(args, "follow", False),
            tail=getattr(args, "tail", 50),
        )
        return

    if args.action == "restart":
        daemon.stop()
        # Re-run in background mode
        argv = sys.argv[:]
        # Replace 'restart' with 'run' and ensure -b is present
        argv = [a if a != "restart" else "run" for a in argv]
        if "-b" not in argv and "--background" not in argv:
            argv.append("-b")
        daemon.start_background(argv)
        return

    # ── Actions requiring instance discovery ──

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
                "× No vLLM instances found. Use -e to specify endpoints manually."
            )
            return

        # Background mode: fork and exit
        if getattr(args, "background", False):
            if daemon.is_running():
                logger.warn(
                    f"[qvl_machine] Daemon already running "
                    f"(PID {daemon.get_pid()}). Use 'restart' to replace it."
                )
                return
            daemon.start_background(sys.argv)
            return

        # Foreground mode: write PID for status checks, run directly
        is_daemon = os.environ.get("_QVL_MACHINE_DAEMON") == "1"
        if is_daemon:
            daemon.write_pid()

        server = QVLMachineServer(
            instances=instances,
            port=args.port,
            timeout=args.timeout,
            enable_perf_tracking=args.perf_track,
        )

        try:
            server.run()
        finally:
            if is_daemon:
                daemon.remove_pid()


if __name__ == "__main__":
    main()
