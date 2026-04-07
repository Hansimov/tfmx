"""TEI (Text Embeddings Inference) Machine Manager

This module provides a load-balanced proxy server that distributes embedding
requests across multiple TEI Docker instances running on different GPUs.
"""

# ANCHOR[id=machine-clis]
CLI_EPILOG = """
Examples:
  # Start machine server (auto-discover TEI containers)
    tei machine run                   # Start on default port 28800 with smart GPU LSH
    tei machine run -p 28800          # Start on specific port
    tei machine run --auto-start      # Auto-start compose backends if none are running
  
  # Filter containers by name pattern
    tei machine run -n "qwen3-embedding"  # Only match containers with this pattern
  
  # Manual endpoint specification (skip auto-discovery)
    tei machine run -e "http://localhost:28880,http://localhost:28881"
  
  # With custom batch size per instance
    tei machine run -b 50             # Max 50 inputs per request to each instance
  
  # LSH computation options
    tei machine run --no-gpu-lsh      # Force CPU for LSH computation
  
  # Performance tracking
    tei machine run --perf-track      # Enable detailed performance tracking
  
  # Check discovered instances without starting server
    tei machine discover              # List all discovered TEI instances
  
  # Health check all instances
    tei machine health                # Check health of all instances
"""

import argparse
import asyncio
import subprocess
import re

import httpx
import numpy as np
import orjson
import uvicorn

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tclogger import logger, logstr
from typing import Callable, Optional, Union
from webu import setup_swagger_ui

from ..utils.lsh import LSHConverter
from ..utils.service_bootstrap import ensure_backend_instances
from ..utils.service_bootstrap import handle_port_conflicts
from .perf_tracker import PerfTracker
from .compose import MAX_CLIENT_BATCH_SIZE, MODEL_NAME as COMPOSE_MODEL_NAME
from .compose import SERVER_PORT as COMPOSE_SERVER_PORT, TEIComposer
from .compose import parse_gpu_configs
from .scheduler import (
    IdleFillingScheduler,
    distribute_with_adaptive_pipeline,
)


PORT = 28800
BATCH_SIZE = MAX_CLIENT_BATCH_SIZE  # Use value from tei_compose
# Probe batch size for adaptive scheduling (small for quick measurement)
MICRO_BATCH_SIZE = 100
MIN_BATCH_SIZE = 50  # Minimum batch size
MAX_BATCH_SIZE = MAX_CLIENT_BATCH_SIZE  # Must match TEI container limit
TEI_CONTAINER_IMAGE_PATTERN = "text-embeddings-inference"
BACKEND_STARTUP_TIMEOUT_SEC = 300.0
BACKEND_STARTUP_POLL_INTERVAL_SEC = 5.0
BACKEND_DISCOVERY_SETTLE_SEC = 10.0
HEALTH_REFRESH_INTERVAL_SEC = 10.0


class TEIBackendRequestError(RuntimeError):
    def __init__(
        self,
        instance: "TEIInstance",
        *,
        status_code: int,
        detail: str,
        error_type: str = "",
    ):
        self.instance = instance
        self.status_code = status_code
        self.detail = detail
        self.error_type = error_type
        super().__init__(f"Instance {instance.port} error: {detail}")

    @property
    def retryable(self) -> bool:
        return self.status_code >= 500 or self.error_type.lower() in {
            "backend",
            "unhealthy",
        }

    @property
    def normalized_detail(self) -> str:
        return self.detail.lower()

    @property
    def capacity_related(self) -> bool:
        return (
            "out of memory" in self.normalized_detail
            or "cuda_error_out_of_memory" in self.normalized_detail
            or "oom" in self.normalized_detail
        )

    @property
    def fatal_backend(self) -> bool:
        return (
            "launch failed" in self.normalized_detail
            or "cuda_error_launch_failed" in self.normalized_detail
            or self.error_type.lower() == "unhealthy"
        )

    @property
    def should_mark_unhealthy(self) -> bool:
        return self.fatal_backend or (
            self.status_code >= 500 and not self.capacity_related
        )


class EmbedRequest(BaseModel):
    """Request model for embedding endpoint."""

    inputs: Union[str, list[str]] = Field(
        ...,
        description="Text or list of texts to embed",
        examples=["Hello, world!", ["Hello", "World"]],
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize embeddings to unit length",
    )
    truncate: bool = Field(
        default=True,
        description="Whether to truncate inputs that exceed max length",
    )


class LSHRequest(EmbedRequest):
    """Request model for LSH endpoint."""

    bitn: int = Field(
        default=2048,
        description="Number of LSH hash bits",
        ge=64,
        le=8192,
    )


class RerankRequest(BaseModel):
    """Request model for rerank endpoint."""

    queries: list[str] = Field(
        ...,
        description="List of query texts to rank passages against",
        examples=[["What is machine learning?", "How does AI work?"]],
    )
    passages: list[str] = Field(
        ...,
        description="List of passage texts to be ranked for each query",
        examples=[
            [
                "Machine learning is a subset of AI.",
                "Deep learning uses neural networks.",
            ]
        ],
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize embeddings to unit length",
    )
    truncate: bool = Field(
        default=True,
        description="Whether to truncate inputs that exceed max length",
    )


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str = Field(..., description="Health status", examples=["healthy"])
    healthy: int = Field(..., description="Number of healthy instances")
    total: int = Field(..., description="Total number of instances")


class InstanceInfo(BaseModel):
    """Information about a single TEI instance."""

    name: str = Field(..., description="Container name")
    endpoint: str = Field(..., description="HTTP endpoint URL")
    gpu_id: Optional[int] = Field(None, description="GPU device ID")
    healthy: bool = Field(..., description="Whether instance is healthy")


class MachineStats(BaseModel):
    """Statistics for the machine."""

    total_requests: int = Field(0, description="Total number of requests processed")
    total_inputs: int = Field(0, description="Total number of inputs embedded")
    total_errors: int = Field(0, description="Total number of errors")
    requests_per_instance: dict[str, int] = Field(
        default_factory=dict, description="Request count per instance"
    )
    # Inter-request gap statistics (in milliseconds)
    inter_request_gap_avg_ms: Optional[float] = Field(
        None, description="Average inter-request gap in milliseconds"
    )
    inter_request_gap_min_ms: Optional[float] = Field(
        None, description="Minimum inter-request gap in milliseconds"
    )
    inter_request_gap_max_ms: Optional[float] = Field(
        None, description="Maximum inter-request gap in milliseconds"
    )
    inter_request_gap_samples: int = Field(
        0, description="Number of inter-request gap samples collected"
    )


class InfoResponse(BaseModel):
    """Response model for info endpoint."""

    port: int = Field(..., description="Machine server port")
    instances: list[InstanceInfo] = Field(..., description="List of TEI instances")
    stats: MachineStats = Field(..., description="Machine statistics")
    scheduler_stats: dict = Field(..., description="Adaptive scheduler statistics")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")


@dataclass
class TEIInstance:
    """Represents a single TEI Docker instance."""

    container_name: str
    host: str
    port: int
    gpu_id: Optional[int] = None
    healthy: bool = False

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def embed_url(self) -> str:
        return f"{self.endpoint}/embed"

    @property
    def health_url(self) -> str:
        return f"{self.endpoint}/health"

    def __repr__(self) -> str:
        status = "✓" if self.healthy else "×"
        gpu_info = f"GPU{self.gpu_id}" if self.gpu_id is not None else "GPU?"
        return (
            f"TEIInstance({status} {self.container_name} @ {self.endpoint}, {gpu_info})"
        )

    def to_info(self) -> InstanceInfo:
        """Convert to InstanceInfo model."""
        return InstanceInfo(
            name=self.container_name,
            endpoint=self.endpoint,
            gpu_id=self.gpu_id,
            healthy=self.healthy,
        )


@dataclass
class TEIMachineStatsData:
    """Statistics for the machine (internal dataclass)."""

    total_requests: int = 0
    total_inputs: int = 0
    total_errors: int = 0
    requests_per_instance: dict = field(default_factory=dict)

    def to_model(
        self, inter_request_gaps: Optional[list[float]] = None
    ) -> MachineStats:
        """Convert to Pydantic model.

        Args:
            inter_request_gaps: List of inter-request gap times in milliseconds
        """
        # Calculate gap statistics if data available
        gap_avg = None
        gap_min = None
        gap_max = None
        gap_samples = 0
        if inter_request_gaps and len(inter_request_gaps) > 0:
            gap_samples = len(inter_request_gaps)
            gap_avg = sum(inter_request_gaps) / gap_samples
            gap_min = min(inter_request_gaps)
            gap_max = max(inter_request_gaps)

        return MachineStats(
            total_requests=self.total_requests,
            total_inputs=self.total_inputs,
            total_errors=self.total_errors,
            requests_per_instance=self.requests_per_instance,
            inter_request_gap_avg_ms=round(gap_avg, 2) if gap_avg else None,
            inter_request_gap_min_ms=round(gap_min, 2) if gap_min else None,
            inter_request_gap_max_ms=round(gap_max, 2) if gap_max else None,
            inter_request_gap_samples=gap_samples,
        )


class TEIInstanceDiscovery:
    """Discovers running TEI Docker instances."""

    @staticmethod
    def discover(name_pattern: Optional[str] = None) -> list[TEIInstance]:
        """
        Discover running TEI Docker containers and their exposed ports.

        Args:
            name_pattern: Optional regex pattern to filter container names

        Returns:
            List of discovered TEIInstance objects
        """
        try:
            if name_pattern:
                # filter by user-specified name pattern
                cmd = f"docker ps --format '{{{{.Names}}}}|{{{{.Image}}}}|{{{{.Ports}}}}' --filter 'name={name_pattern}'"
            else:
                # get all containers and filter by image name
                # note: 'ancestor' filter with wildcards doesn't work reliably
                cmd = "docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}'"

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                logger.warn(f"× Docker command failed: {result.stderr}")
                return []

            if not result.stdout.strip():
                logger.note(f"[tei_machine] No running containers found")
                return []

            instances = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split("|")
                if len(parts) < 3:
                    continue

                container_name, image, ports = parts[0], parts[1], parts[2]

                # Filter by image containing TEI pattern
                if TEI_CONTAINER_IMAGE_PATTERN not in image:
                    continue

                # Filter by name pattern if specified (when not using name filter in docker ps)
                if name_pattern and not re.search(name_pattern, container_name):
                    continue

                # Extract host port from port mapping or container inspect (for host network mode)
                host_port = TEIInstanceDiscovery._extract_host_port(
                    ports, container_name
                )
                if host_port is None:
                    continue

                # Extract GPU ID from container name (e.g., "tei--xxx--gpu0" -> 0)
                gpu_id = TEIInstanceDiscovery._extract_gpu_id(container_name)

                instance = TEIInstance(
                    container_name=container_name,
                    host="localhost",
                    port=host_port,
                    gpu_id=gpu_id,
                )
                instances.append(instance)

            # Sort by GPU ID for consistent ordering
            instances.sort(key=lambda x: (x.gpu_id if x.gpu_id is not None else 999))

            # if instances:
            #     logger.okay(f"[tei_machine] Found {len(instances)} TEI containers")
            #     for inst in instances:
            #         logger.mesg(f"  - {inst.container_name} @ {inst.endpoint}")

            return instances

        except Exception as e:
            logger.warn(f"× Failed to discover TEI instances: {e}")
            return []

    @staticmethod
    def _extract_host_port(ports_str: str, container_name: str = "") -> Optional[int]:
        """Extract host port from Docker port mapping string or container inspect.

        For bridge mode: Parse ports_str (e.g., "0.0.0.0:28880->80/tcp")
        For host mode: Extract --port argument from container's command
        """
        # Try bridge mode first (port mapping exists)
        match = re.search(r"(?:0\.0\.0\.0|::):(\d+)->", ports_str)
        if match:
            return int(match.group(1))

        # Fallback for host network mode (empty ports_str)
        if not ports_str and container_name:
            try:
                # Get container's Args field and extract --port value
                result = subprocess.run(
                    ["docker", "inspect", container_name, "--format", "{{.Args}}"],
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
        """Extract GPU ID from container name."""
        # Match patterns like "--gpu0", "--gpu1", etc.
        match = re.search(r"--gpu(\d+)", container_name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def from_endpoints(endpoints: list[str]) -> list[TEIInstance]:
        """Create TEIInstance list from manual endpoint specifications."""
        instances = []
        for i, endpoint in enumerate(endpoints):
            # Parse endpoint URL
            endpoint = endpoint.strip()
            if not endpoint:
                continue

            # Extract host and port
            match = re.match(r"https?://([^:]+):(\d+)", endpoint)
            if match:
                host, port = match.group(1), int(match.group(2))
            else:
                # Assume localhost if only port specified
                try:
                    port = int(endpoint)
                    host = "localhost"
                except ValueError:
                    continue

            instance = TEIInstance(
                container_name=f"manual-{i}",
                host=host,
                port=port,
                gpu_id=i,
            )
            instances.append(instance)

        return instances


class LSHConverterCache:
    """Cache for LSHConverter instances to avoid repeated initialization.

    Automatically uses GPU acceleration if available.
    """

    def __init__(self, use_gpu: bool = True):
        self._cache: dict[tuple[int, int], "LSHConverter"] = {}
        self._lock = asyncio.Lock()
        self.use_gpu = use_gpu

    def get(self, dims: int, bitn: int) -> "LSHConverter":
        """Get or create LSHConverter for given dimensions and bit count.

        Uses GPU acceleration by default if available.
        """
        key = (dims, bitn)
        if key not in self._cache:
            self._cache[key] = LSHConverter(
                dims=dims,
                bitn=bitn,
                verbose=False,
                use_gpu=self.use_gpu,
            )
        return self._cache[key]


class TEIMachineServer:
    """FastAPI server that proxies requests to multiple TEI instances."""

    def __init__(
        self,
        instances: list[TEIInstance],
        port: int = PORT,
        batch_size: int = BATCH_SIZE,
        micro_batch_size: int = MICRO_BATCH_SIZE,
        timeout: float = 60.0,
        use_gpu_lsh: bool = True,
        enable_perf_tracking: bool = False,
        batch_wait_ms: float = 5.0,  # Time to wait for more requests before processing
        discover_instances_fn: Callable[[], list[TEIInstance]] | None = None,
    ):
        self.instances = instances
        self.port = port
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.timeout = timeout
        self._discover_instances_fn = discover_instances_fn
        self.stats = TEIMachineStatsData()
        self._client: Optional[httpx.AsyncClient] = None
        self._health_task: Optional[asyncio.Task] = None
        self._lsh_cache = LSHConverterCache(use_gpu=use_gpu_lsh)

        # Performance tracking
        self.enable_perf_tracking = enable_perf_tracking
        self.perf_tracker = PerfTracker(
            name="tei_machine", verbose=enable_perf_tracking
        )
        self._request_counter = 0

        # Inter-request ARRIVAL gap tracking (time between consecutive request arrivals)
        self._last_arrival_time: Optional[float] = None
        self._inter_request_gaps: list[float] = []  # in milliseconds
        self._gap_window_size: int = 100  # Keep last N gaps for rolling stats

        # Shared scheduler for load balancing across requests
        self.scheduler = IdleFillingScheduler(
            workers=instances,
            get_worker_id=lambda inst: inst.container_name,
            max_batch_size=batch_size,
        )
        self._instance_capacity_limits: dict[str, int] = {}

        # Lock for serializing GPU access - multiple concurrent requests would
        # compete for the same GPUs, causing severe performance degradation
        self._scheduler_lock: Optional[asyncio.Lock] = None
        self._pending_requests: int = 0  # Counter for requests waiting/processing

        # Request batching: collect multiple requests and process together
        self._batch_wait_ms = batch_wait_ms
        self._batch_queue: list[tuple[list[str], asyncio.Future]] = []
        self._batch_lock: Optional[asyncio.Lock] = None
        self._batch_processing: bool = False

        # Create FastAPI app
        self.app = self._create_app()

    def get_healthy_instances(self) -> list[TEIInstance]:
        """Get all healthy instances."""
        return [i for i in self.instances if i.healthy]

    @staticmethod
    def _instance_identity(instance: TEIInstance) -> str:
        return instance.container_name or instance.endpoint

    @staticmethod
    def _instance_sort_key(instance: TEIInstance) -> tuple[int, str]:
        return (
            instance.gpu_id if instance.gpu_id is not None else 999,
            instance.container_name,
        )

    def _update_scheduler_workers(self) -> None:
        self.scheduler.update_workers(self.get_healthy_instances())

    def _get_instance_capacity_limit(self, instance: TEIInstance) -> int | None:
        return self._instance_capacity_limits.get(self._instance_identity(instance))

    def _record_instance_capacity_limit(
        self,
        instance: TEIInstance,
        *,
        attempted_size: int,
        inferred_limit: int,
    ) -> None:
        normalized_limit = max(1, inferred_limit)
        identity = self._instance_identity(instance)
        current_limit = self._instance_capacity_limits.get(identity)

        if current_limit is not None and normalized_limit >= current_limit:
            return

        self._instance_capacity_limits[identity] = normalized_limit
        logger.warn(
            f"[tei_machine] Learned overload-safe batch limit for {instance.container_name}: <= {normalized_limit} after failure at {attempted_size}"
        )

    async def _send_embed_request_with_capacity_limit(
        self,
        instance: TEIInstance,
        inputs: list[str],
        normalize: bool,
        truncate: bool,
    ) -> np.ndarray:
        capacity_limit = self._get_instance_capacity_limit(instance)
        if capacity_limit is None or len(inputs) <= capacity_limit:
            return await self._send_embed_request_np(
                instance,
                inputs,
                normalize,
                truncate,
            )

        arrays = []
        start = 0
        while start < len(inputs):
            current_limit = (
                self._get_instance_capacity_limit(instance) or capacity_limit
            )
            arrays.append(
                await self._send_embed_request_np(
                    instance,
                    inputs[start : start + current_limit],
                    normalize,
                    truncate,
                )
            )
            start += current_limit

        if len(arrays) == 1:
            return arrays[0]
        return np.vstack(arrays)

    def _merge_discovered_instances(
        self,
        discovered: list[TEIInstance],
    ) -> tuple[bool, list[TEIInstance]]:
        existing_by_identity = {
            self._instance_identity(instance): instance for instance in self.instances
        }
        changed = False
        added_instances: list[TEIInstance] = []

        for discovered_instance in discovered:
            identity = self._instance_identity(discovered_instance)
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
            ):
                existing.host = discovered_instance.host
                existing.port = discovered_instance.port
                existing.gpu_id = discovered_instance.gpu_id
                changed = True

        if changed:
            self.instances.sort(key=self._instance_sort_key)
        return changed, added_instances

    async def _refresh_instances_from_discovery(self) -> None:
        if self._discover_instances_fn is None:
            return

        discovered = await asyncio.to_thread(self._discover_instances_fn)
        if not discovered:
            return

        changed, added_instances = self._merge_discovered_instances(discovered)
        if not changed:
            return

        for instance in added_instances:
            logger.okay(
                f"[tei_machine] Added backend instance: {instance.container_name} @ {instance.endpoint}"
            )
        self._update_scheduler_workers()

    def _is_retryable_upstream_status(self, status_code: int) -> bool:
        return status_code >= 500

    def _is_retryable_upstream_exception(self, exc: Exception) -> bool:
        if isinstance(exc, TEIBackendRequestError):
            return exc.retryable
        return isinstance(
            exc,
            (
                httpx.TransportError,
                httpx.TimeoutException,
                asyncio.TimeoutError,
            ),
        )

    def _mark_instance_unhealthy(
        self,
        instance: TEIInstance,
        reason: Exception | str,
    ) -> None:
        reason_text = str(reason)
        if instance.healthy:
            logger.warn(
                f"× Marking {instance.container_name} unhealthy for TEI failover: {reason_text}"
            )
        instance.healthy = False
        self._update_scheduler_workers()

    @staticmethod
    def _extract_backend_error(response: httpx.Response) -> tuple[str, str]:
        detail = response.text
        error_type = ""
        try:
            payload = response.json()
            detail = payload.get("error", detail)
            error_type = payload.get("error_type", "")
        except Exception:
            pass
        return detail, error_type

    async def _run_embeddings_with_failover(
        self,
        inputs: list[str],
        normalize: bool,
        truncate: bool,
    ) -> np.ndarray:
        max_attempts = max(1, len(self.instances))
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            healthy = self.get_healthy_instances()
            if not healthy:
                if last_error is not None:
                    raise last_error
                raise HTTPException(
                    status_code=503,
                    detail="No healthy instances available",
                )

            before_attempt = {instance.container_name for instance in healthy}
            try:
                return await self._distribute_with_scheduler_np(
                    inputs,
                    healthy,
                    normalize,
                    truncate,
                )
            except Exception as exc:
                last_error = exc
                after_attempt = {
                    instance.container_name for instance in self.get_healthy_instances()
                }
                lost_instances = before_attempt - after_attempt
                if lost_instances and after_attempt and attempt < max_attempts:
                    logger.warn(
                        f"[tei_machine] Retrying request after backend failures on {sorted(lost_instances)}; remaining healthy instances: {len(after_attempt)}/{len(self.instances)}"
                    )
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise HTTPException(status_code=503, detail="No healthy instances available")

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="TEI Machine",
            description="Load-balanced proxy for Text Embeddings Inference instances",
            version="1.0.0",
            lifespan=self._lifespan,
            docs_url=None,  # Disable default docs
            redoc_url=None,  # Disable redoc
        )

        # Setup custom Swagger UI
        setup_swagger_ui(app)

        # Register routes
        app.get(
            "/health",
            response_model=HealthResponse,
            summary="Health check",
            description="Check health status of the machine",
        )(self.health)

        app.get(
            "/info",
            response_model=InfoResponse,
            summary="Machine info",
            description="Get detailed information about the machine and statistics",
        )(self.info)

        app.post(
            "/embed",
            response_model=list[list[float]],
            summary="Generate embeddings",
            description="Generate embeddings for input texts using load-balanced TEI instances",
        )(self.embed)

        app.post(
            "/lsh",
            response_model=list[str],
            summary="Generate LSH hashes",
            description="Generate LSH hash hex strings for input texts (embed + LSH)",
        )(self.lsh)

        app.post(
            "/rerank",
            response_model=list[list[tuple[int, float]]],
            summary="Rerank passages for queries",
            description="Compute cosine similarity between queries and passages, return sorted rankings",
        )(self.rerank)

        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage application lifecycle."""
        # Startup
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))

        # Initialize the scheduler lock (must be done in async context)
        self._scheduler_lock = asyncio.Lock()

        # Initialize batch lock
        self._batch_lock = asyncio.Lock()

        # Initial health check
        await self.health_check_all()

        healthy_instances = self.get_healthy_instances()
        if not healthy_instances:
            logger.warn("× No healthy TEI instances available at startup")

        # Start background health checker
        self._health_task = asyncio.create_task(self._periodic_health_check())

        logger.okay(f"[tei_machine] Started on port {self.port}")
        healthy_str = logstr.okay(len(healthy_instances))
        total_str = logstr.mesg(len(self.instances))
        logger.mesg(f"[tei_machine] Healthy instances: {healthy_str}/{total_str}")

        yield

        # Shutdown
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()

    async def _periodic_health_check(self) -> None:
        """Periodically check health of all instances."""
        while True:
            await asyncio.sleep(HEALTH_REFRESH_INTERVAL_SEC)
            await self.health_check_all()

    async def health_check_all(self) -> None:
        """Check health of all instances."""
        await self._refresh_instances_from_discovery()
        if not self._client:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))

        tasks = [self._check_instance_health(inst) for inst in self.instances]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._update_scheduler_workers()

    async def _check_instance_health(self, instance: TEIInstance) -> bool:
        """Check health of a single instance."""
        try:
            resp = await self._client.get(instance.health_url)
            instance.healthy = resp.status_code == 200
            return instance.healthy
        except Exception:
            instance.healthy = False
            return False

    async def embed(self, request: EmbedRequest) -> list[list[float]]:
        """Handle embedding requests with load balancing and batching."""
        # Normalize inputs to list
        inputs = request.inputs
        if isinstance(inputs, str):
            inputs = [inputs]

        if not inputs:
            raise HTTPException(status_code=400, detail="No inputs provided")

        self.stats.total_requests += 1
        self.stats.total_inputs += len(inputs)

        # Get healthy instances
        healthy = self.get_healthy_instances()
        if not healthy:
            self.stats.total_errors += 1
            raise HTTPException(
                status_code=503, detail="No healthy instances available"
            )

        try:
            # Use lock to serialize GPU access
            async with self._scheduler_lock:
                # Use optimized numpy path for all embedding operations
                embs_array = await self._run_embeddings_with_failover(
                    inputs,
                    request.normalize,
                    request.truncate,
                )
            # Convert numpy array to list for JSON response
            # tolist() is faster than manually rebuilding nested lists
            return embs_array.tolist()

        except Exception as e:
            self.stats.total_errors += 1
            logger.warn(f"× Embed error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _send_embed_request_np(
        self,
        instance: TEIInstance,
        inputs: list[str],
        normalize: bool,
        truncate: bool,
    ) -> np.ndarray:
        """Send embedding request and return numpy array directly.

        Optimized path: uses orjson for fast JSON parsing and converts
        to numpy per-chunk, enabling overlap with network I/O of other chunks.
        """
        payload = {
            "inputs": inputs,
            "normalize": normalize,
            "truncate": truncate,
        }

        try:
            resp = await self._client.post(instance.embed_url, json=payload)
        except Exception as exc:
            if self._is_retryable_upstream_exception(exc):
                self._mark_instance_unhealthy(instance, exc)
            raise

        if resp.status_code != 200:
            detail, error_type = self._extract_backend_error(resp)
            error = TEIBackendRequestError(
                instance,
                status_code=resp.status_code,
                detail=detail,
                error_type=error_type,
            )
            if error.capacity_related and len(inputs) > 1:
                midpoint = max(1, len(inputs) // 2)
                self._record_instance_capacity_limit(
                    instance,
                    attempted_size=len(inputs),
                    inferred_limit=max(midpoint, len(inputs) - midpoint),
                )
                left = await self._send_embed_request_np(
                    instance,
                    inputs[:midpoint],
                    normalize,
                    truncate,
                )
                right = await self._send_embed_request_np(
                    instance,
                    inputs[midpoint:],
                    normalize,
                    truncate,
                )
                return np.vstack([left, right])
            if error.should_mark_unhealthy:
                self._mark_instance_unhealthy(instance, error)
            raise error

        # Parse JSON with orjson (10x faster) and convert to numpy array directly
        # orjson.loads returns Python list, which we convert to numpy
        # We do this per-chunk so it can overlap with network I/O of other chunks
        data = orjson.loads(resp.content)
        return np.array(data, dtype=np.float32)

    async def _distribute_with_scheduler_np(
        self,
        inputs: list[str],
        instances: list[TEIInstance],
        normalize: bool,
        truncate: bool,
    ) -> np.ndarray:
        """
        Distribute inputs using scheduler with numpy optimization.

        Returns numpy array directly instead of list[list[float]].
        Each chunk is converted to numpy immediately after receiving,
        avoiding the expensive nested-list-to-array conversion.

        Used by both /embed and /lsh endpoints for optimal performance.
        """
        # Update scheduler with current healthy instances
        self.scheduler.update_workers(instances)

        # Define the async process function that returns numpy arrays
        async def process_on_instance_np(
            instance: TEIInstance, chunk: list[str]
        ) -> np.ndarray:
            result = await self._send_embed_request_with_capacity_limit(
                instance, chunk, normalize, truncate
            )
            # Update stats
            instance_name = instance.container_name
            self.stats.requests_per_instance[instance_name] = (
                self.stats.requests_per_instance.get(instance_name, 0) + 1
            )
            return result

        # Use adaptive pipeline scheduling (optimal for heterogeneous GPUs)
        embeddings_list, details = await distribute_with_adaptive_pipeline(
            scheduler=self.scheduler,
            inputs=inputs,
            process_func=process_on_instance_np,
            enable_perf_tracking=self.enable_perf_tracking,
            perf_tracker=self.perf_tracker,
            min_batch_size=MIN_BATCH_SIZE,
            max_batch_size=MAX_BATCH_SIZE,
            probe_batch_size=self.micro_batch_size,
        )

        # Print performance analysis periodically
        if self.enable_perf_tracking:
            self._request_counter += 1

        # Combine numpy arrays efficiently using vstack
        # embeddings_list is a list of numpy arrays (one per chunk)
        # np.vstack is much faster than np.array(nested_list)
        if len(embeddings_list) == 1:
            return embeddings_list[0]
        else:
            return np.vstack(embeddings_list)

    async def health(self) -> HealthResponse:
        """Handle health check requests."""
        healthy_instances = self.get_healthy_instances()
        healthy_count = len(healthy_instances)
        total_count = len(self.instances)

        response = HealthResponse(
            status="healthy" if healthy_count > 0 else "unhealthy",
            healthy=healthy_count,
            total=total_count,
        )

        if healthy_count == 0:
            raise HTTPException(
                status_code=503,
                detail=response.model_dump(),
            )

        return response

    async def info(self) -> InfoResponse:
        """Handle info requests."""
        return InfoResponse(
            port=self.port,
            instances=[inst.to_info() for inst in self.instances],
            stats=self.stats.to_model(inter_request_gaps=self._inter_request_gaps),
            scheduler_stats=self.scheduler.get_stats_summary(),
        )

    async def lsh(self, request: LSHRequest) -> list[str]:
        """Handle LSH requests: embed + LSH conversion to hex strings.

        Uses dynamic batching: requests arriving while another is processing
        will be queued and processed together in the next batch.
        """
        import time as _time

        t0 = _time.perf_counter()

        # Calculate inter-request ARRIVAL gap (time since last request ARRIVED)
        # This measures how frequently requests are coming in, regardless of processing
        inter_request_gap_ms: Optional[float] = None
        if self._last_arrival_time is not None:
            inter_request_gap_ms = (t0 - self._last_arrival_time) * 1000
            self._inter_request_gaps.append(inter_request_gap_ms)
            # Keep only last N gaps for rolling stats
            if len(self._inter_request_gaps) > self._gap_window_size:
                self._inter_request_gaps.pop(0)
        # Update arrival time IMMEDIATELY (before processing)
        self._last_arrival_time = t0

        # Track pending requests (for monitoring queue depth)
        self._pending_requests += 1

        # Normalize inputs to list
        inputs = request.inputs
        if isinstance(inputs, str):
            inputs = [inputs]

        if not inputs:
            self._pending_requests -= 1
            raise HTTPException(status_code=400, detail="No inputs provided")

        self.stats.total_requests += 1
        self.stats.total_inputs += len(inputs)

        # Get healthy instances
        healthy = self.get_healthy_instances()
        if not healthy:
            self._pending_requests -= 1
            self.stats.total_errors += 1
            raise HTTPException(
                status_code=503, detail="No healthy instances available"
            )

        try:
            # Use lock to serialize GPU access - concurrent requests would compete
            # for GPUs and cause severe performance degradation
            async with self._scheduler_lock:
                t1 = _time.perf_counter()

                # Use optimized numpy distribution that returns numpy array directly
                # This avoids the expensive nested-list-to-array conversion
                embs_array = await self._run_embeddings_with_failover(
                    inputs,
                    request.normalize,
                    request.truncate,
                )

                t2 = _time.perf_counter()

            # embs_array is already numpy array, just get dims
            dims = embs_array.shape[1]

            t3 = _time.perf_counter()

            # Get cached LSH converter
            lsh: LSHConverter = self._lsh_cache.get(dims=dims, bitn=request.bitn)

            # Vectorized conversion to hex strings
            lsh_hashes = lsh.embs_to_hex_batch(embs_array)

            t4 = _time.perf_counter()

            # Note: _last_arrival_time is updated at REQUEST ARRIVAL (not end)
            # This gives us inter-arrival time instead of inter-completion time

            # Log detailed timing if perf tracking enabled
            if self.enable_perf_tracking:
                total = (t4 - t0) * 1000
                embed_time = (t2 - t1) * 1000
                # t3 - t2 is now minimal (just extracting dims from existing array)
                np_overhead = (t3 - t2) * 1000
                lsh_time = (t4 - t3) * 1000

                # Build gap info string - this is now ARRIVAL gap
                gap_info = ""
                if inter_request_gap_ms is not None:
                    gap_info = f", arrival_gap={inter_request_gap_ms:.1f}ms"
                    # Add rolling average if we have enough samples
                    if len(self._inter_request_gaps) >= 3:
                        avg_gap = sum(self._inter_request_gaps) / len(
                            self._inter_request_gaps
                        )
                        gap_info += f" (avg={avg_gap:.1f}ms)"

                logger.mesg(
                    f"[LSH timing] n={len(inputs)}, total={total:.1f}ms | "
                    f"embed={embed_time:.1f}ms, lsh={lsh_time:.1f}ms{gap_info}"
                )

            self._pending_requests -= 1
            return lsh_hashes

        except Exception as e:
            self._pending_requests -= 1
            self.stats.total_errors += 1
            logger.warn(f"× LSH error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def rerank(self, request: RerankRequest) -> list[list[tuple[int, float]]]:
        """Rerank passages for each query based on cosine similarity.

        Computes embeddings for all passages and queries, then calculates
        cosine similarity between each query and all passages. Returns
        rankings for each query, preserving the original passage order.

        Args:
            request: RerankRequest with queries and passages

        Returns:
            List of rankings for each query. Each ranking is a list of
            (rank_position, similarity_score) tuples in passage input order.
            rank_position: 0 = best match (highest similarity).
        """
        queries = request.queries
        passages = request.passages

        if not queries:
            raise HTTPException(status_code=400, detail="No queries provided")
        if not passages:
            raise HTTPException(status_code=400, detail="No passages provided")

        self.stats.total_requests += 1
        self.stats.total_inputs += len(queries) + len(passages)

        # Get healthy instances
        healthy = self.get_healthy_instances()
        if not healthy:
            self.stats.total_errors += 1
            raise HTTPException(
                status_code=503, detail="No healthy instances available"
            )

        try:
            async with self._scheduler_lock:
                # Embed all texts together for efficiency
                # Combine queries and passages: [queries..., passages...]
                all_texts = queries + passages
                all_embs = await self._run_embeddings_with_failover(
                    all_texts,
                    request.normalize,
                    request.truncate,
                )

                # Split embeddings back into queries and passages
                n_queries = len(queries)
                query_embs = all_embs[:n_queries]  # shape: (n_queries, dims)
                passage_embs = all_embs[n_queries:]  # shape: (n_passages, dims)

                # Compute cosine similarity matrix: (n_queries, n_passages)
                # Since embeddings are already normalized (normalize=True by default),
                # cosine similarity is just the dot product
                # For non-normalized case, we normalize here
                if not request.normalize:
                    # Normalize for cosine similarity
                    query_norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
                    passage_norms = np.linalg.norm(passage_embs, axis=1, keepdims=True)
                    query_embs = query_embs / np.maximum(query_norms, 1e-8)
                    passage_embs = passage_embs / np.maximum(passage_norms, 1e-8)

                # Compute similarity matrix: (n_queries, n_passages)
                sim_matrix = np.dot(query_embs, passage_embs.T)

                # For each query, create rankings preserving passage order
                # Sort indices by similarity (descending) to get rank positions
                sorted_indices = np.argsort(-sim_matrix, axis=1)

                # Build result: list[list[tuple[int, float]]]
                # Each inner list preserves passage input order
                # tuple is (rank_position, similarity_score)
                # rank_position: where this passage ranks (0=best match)
                results = []
                n_passages = len(passages)
                for q_idx in range(n_queries):
                    # Create rank lookup: passage_idx -> rank_position
                    rank_lookup = np.empty(n_passages, dtype=np.int32)
                    for rank, p_idx in enumerate(sorted_indices[q_idx]):
                        rank_lookup[p_idx] = rank

                    # Build ranking list in passage input order
                    query_rankings = []
                    for p_idx in range(n_passages):
                        rank_pos = int(rank_lookup[p_idx])
                        sim_score = float(sim_matrix[q_idx, p_idx])
                        query_rankings.append((rank_pos, sim_score))
                    results.append(query_rankings)

            return results

        except Exception as e:
            self.stats.total_errors += 1
            logger.warn(f"× Rerank error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def run(self) -> None:
        """Run the server using uvicorn."""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info",
        )

    async def run_async(self) -> None:
        """Run the server asynchronously."""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Run the TEI machine load-balanced proxy"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = CLI_EPILOG
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
        help="Regex pattern to filter container names",
    )
    parser.add_argument(
        "-e",
        "--endpoints",
        type=str,
        default=None,
        help="Comma-separated list of TEI endpoints (skip auto-discovery)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Max batch size per instance (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "-m",
        "--micro-batch-size",
        type=int,
        default=MICRO_BATCH_SIZE,
        help=f"Micro-batch size for pipeline scheduling (default: {MICRO_BATCH_SIZE})",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--on-conflict",
        choices=["report", "replace"],
        default="report",
        help="How to handle an existing machine listener on the target port",
    )
    parser.add_argument(
        "--no-gpu-lsh",
        action="store_true",
        help="Disable GPU acceleration for LSH computation (use CPU instead)",
    )
    parser.add_argument(
        "--perf-track",
        action="store_true",
        help="Enable detailed performance tracking to identify bottlenecks",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help=(
            "When no TEI containers are running, start a compose deployment in the "
            "background and wait for healthy backends"
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
            "Compose backend base port for auto-started TEI containers "
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
        "--compose-gpu-configs",
        type=str,
        default=None,
        help='Per-GPU config for auto-started backends: "GPU[:MODEL],..."',
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=["run", "discover", "health"],
        default=None,
        help="Action to perform: run, discover, or health",
    )


def _discover_instances_from_docker(name_pattern: Optional[str]) -> list[TEIInstance]:
    return TEIInstanceDiscovery.discover(name_pattern)


def _build_auto_start_composer(args: argparse.Namespace) -> TEIComposer:
    composer_kwargs = {}
    if getattr(args, "compose_model_name", None):
        composer_kwargs["model_name"] = args.compose_model_name
    if getattr(args, "compose_port", None) is not None:
        composer_kwargs["port"] = args.compose_port
    if getattr(args, "compose_project_name", None):
        composer_kwargs["project_name"] = args.compose_project_name
    if getattr(args, "compose_gpus", None):
        composer_kwargs["gpu_ids"] = args.compose_gpus
    if getattr(args, "compose_gpu_configs", None):
        composer_kwargs["gpu_configs"] = parse_gpu_configs(args.compose_gpu_configs)
    return TEIComposer(**composer_kwargs)


def discover_instances(args: argparse.Namespace) -> list[TEIInstance]:
    """Discover or create TEI instances based on args."""
    if args.endpoints:
        endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
        instances = TEIInstanceDiscovery.from_endpoints(endpoints)
        logger.okay(f"[tei_machine] Using {len(instances)} manual endpoints")
        return instances

    instances = _discover_instances_from_docker(args.name_pattern)
    instances = ensure_backend_instances(
        instances,
        enabled=getattr(args, "auto_start", False),
        manual_endpoints=bool(getattr(args, "endpoints", None)),
        service_label="[tei_machine]",
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
    logger.okay(f"[tei_machine] Discovered {len(instances)} TEI instances")
    return instances


def log_instances(instances: list[TEIInstance], show_health: bool = False) -> None:
    """Print discovered instances.

    Args:
        instances: List of TEI instances to display
        show_health: Whether to include health status column
    """
    if not instances:
        logger.warn("× No TEI instances found")
        return

    dash_len = 85
    logger.note("=" * dash_len)

    if show_health:
        logger.note(f"{'GPU':<6} {'CONTAINER':<40} {'ENDPOINT':<25} {'STATUS':<8}")
    else:
        logger.note(f"{'GPU':<6} {'CONTAINER':<45} {'ENDPOINT':<25}")

    logger.note("-" * dash_len)

    for inst in instances:
        gpu_str = str(inst.gpu_id) if inst.gpu_id is not None else "?"
        if show_health:
            if inst.healthy:
                status = logstr.okay("✓ healthy")
            else:
                status = logstr.erro("× sick")
            logger.mesg(
                f"{gpu_str:<6} {inst.container_name:<40} {inst.endpoint:<25} {status:<8}"
            )
        else:
            logger.mesg(f"{gpu_str:<6} {inst.container_name:<45} {inst.endpoint:<25}")

    logger.note("=" * dash_len)

    if show_health:
        healthy = sum(1 for i in instances if i.healthy)
        logger.mesg(f"[tei_machine] Healthy: {healthy}/{len(instances)}")


async def check_health(instances: list[TEIInstance]) -> None:
    """Check health of all instances and print status."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        for inst in instances:
            try:
                resp = await client.get(inst.health_url)
                inst.healthy = resp.status_code == 200
            except Exception:
                inst.healthy = False

    log_instances(instances, show_health=True)


def run_from_args(args: argparse.Namespace) -> None:
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
                "× No TEI instances found. Use -e to specify endpoints manually."
            )
            return

        if not handle_port_conflicts(
            args.port,
            policy=getattr(args, "on_conflict", "report"),
            label="[tei_machine]",
        ):
            return

        if args.perf_track:
            logger.note("[tei_machine] Performance tracking ENABLED")
            logger.note(
                "[tei_machine] Detailed metrics will be logged for each request"
            )

        logger.note(
            f"[tei_machine] Adaptive pipeline scheduling (probe_batch_size={args.micro_batch_size})"
        )

        server = TEIMachineServer(
            instances=instances,
            port=args.port,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            timeout=args.timeout,
            use_gpu_lsh=not args.no_gpu_lsh,
            enable_perf_tracking=args.perf_track,
            discover_instances_fn=(
                None
                if args.endpoints
                else lambda: _discover_instances_from_docker(args.name_pattern)
            ),
        )
        server.run()


class TEIMachineArgParser:
    """Compatibility wrapper around the reusable machine parser."""

    def __init__(self, argv: list[str] | None = None):
        self.parser = argparse.ArgumentParser()
        configure_parser(self.parser)
        self.args = self.parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    arg_parser = TEIMachineArgParser(argv)
    args = arg_parser.args

    if args.action is None:
        arg_parser.parser.print_help()
        return

    run_from_args(args)


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_machine.py#machine-clis
