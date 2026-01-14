"""TEI (Text Embeddings Inference) Machine Manager

This module provides a load-balanced proxy server that distributes embedding
requests across multiple TEI Docker instances running on different GPUs.
"""

# ANCHOR[id=machine-clis]
CLI_EPILOG = """
Examples:
  # Start machine server (auto-discover TEI containers)
  tei_machine run                   # Start on default port 28800 with smart GPU LSH
  tei_machine run -p 28800          # Start on specific port
  
  # Filter containers by name pattern
  tei_machine run -n "qwen3-embedding"  # Only match containers with this pattern
  
  # Manual endpoint specification (skip auto-discovery)
  tei_machine run -e "http://localhost:28880,http://localhost:28881"
  
  # With custom batch size per instance
  tei_machine run -b 50             # Max 50 inputs per request to each instance
  
  # LSH computation options
  tei_machine run --no-gpu-lsh      # Force CPU for LSH computation
  
  # Performance tracking
  tei_machine run --perf-track      # Enable detailed performance tracking
  
  # Check discovered instances without starting server
  tei_machine discover              # List all discovered TEI instances
  
  # Health check all instances
  tei_machine health                # Check health of all instances
"""

import argparse
import asyncio
import subprocess
import re

import httpx
import numpy as np
import uvicorn

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tclogger import logger, logstr
from typing import Optional, Union
from webu import setup_swagger_ui

from .lsh import LSHConverter
from .perf_tracker import PerfTracker
from .tei_compose import MAX_CLIENT_BATCH_SIZE
from .tei_scheduler import (
    IdleFillingScheduler,
    distribute_with_scheduler,
    distribute_with_pipeline,
    distribute_with_adaptive_pipeline,
)


PORT = 28800
BATCH_SIZE = MAX_CLIENT_BATCH_SIZE  # Use value from tei_compose
# Probe batch size for adaptive scheduling (small for quick measurement)
MICRO_BATCH_SIZE = 100
MIN_BATCH_SIZE = 50  # Minimum batch size
MAX_BATCH_SIZE = MAX_CLIENT_BATCH_SIZE  # Must match TEI container limit
TEI_CONTAINER_IMAGE_PATTERN = "text-embeddings-inference"


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

                # Extract host port from port mapping (e.g., "0.0.0.0:28880->80/tcp")
                host_port = TEIInstanceDiscovery._extract_host_port(ports)
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
    def _extract_host_port(ports_str: str) -> Optional[int]:
        """Extract host port from Docker port mapping string."""
        # Match patterns like "0.0.0.0:28880->80/tcp" or ":::28880->80/tcp"
        match = re.search(r"(?:0\.0\.0\.0|::):(\d+)->", ports_str)
        if match:
            return int(match.group(1))
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
        use_pipeline: bool = True,  # Use pipeline scheduling by default
        batch_wait_ms: float = 5.0,  # Time to wait for more requests before processing
    ):
        self.instances = instances
        self.port = port
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.timeout = timeout
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

        # Pipeline mode (eliminates round barrier)
        self.use_pipeline = use_pipeline

        # Shared scheduler for load balancing across requests
        self.scheduler = IdleFillingScheduler(
            workers=instances,
            get_worker_id=lambda inst: inst.container_name,
            max_batch_size=batch_size,
        )

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
            await asyncio.sleep(30)
            await self.health_check_all()

    async def health_check_all(self) -> None:
        """Check health of all instances."""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))

        tasks = [self._check_instance_health(inst) for inst in self.instances]
        await asyncio.gather(*tasks, return_exceptions=True)

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
                embeddings = await self._distribute_with_scheduler(
                    inputs, healthy, request.normalize, request.truncate
                )
            return embeddings

        except Exception as e:
            self.stats.total_errors += 1
            logger.warn(f"× Embed error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _send_embed_request(
        self,
        instance: TEIInstance,
        inputs: list[str],
        normalize: bool,
        truncate: bool,
    ) -> list[list[float]]:
        """Send embedding request to a single instance."""
        payload = {
            "inputs": inputs,
            "normalize": normalize,
            "truncate": truncate,
        }

        resp = await self._client.post(instance.embed_url, json=payload)
        if resp.status_code != 200:
            raise ValueError(f"Instance {instance.port} error: {resp.text}")
        return resp.json()

    async def _distribute_with_scheduler(
        self,
        inputs: list[str],
        instances: list[TEIInstance],
        normalize: bool,
        truncate: bool,
    ) -> list[list[float]]:
        """
        Distribute inputs using scheduler.

        Uses adaptive pipeline by default for optimal heterogeneous GPU utilization,
        or falls back to fixed pipeline/round-based scheduling if disabled.

        NOTE: Uses shared scheduler with lock protection. Multiple concurrent
        requests competing for GPUs causes severe performance degradation.
        """
        # Update scheduler with current healthy instances
        self.scheduler.update_workers(instances)

        # Define the async process function for scheduler
        async def process_on_instance(
            instance: TEIInstance, chunk: list[str]
        ) -> list[list[float]]:
            result = await self._send_embed_request(
                instance, chunk, normalize, truncate
            )
            # Update stats
            instance_name = instance.container_name
            self.stats.requests_per_instance[instance_name] = (
                self.stats.requests_per_instance.get(instance_name, 0) + 1
            )
            return result

        # Use adaptive pipeline, fixed pipeline, or round-based scheduling
        if self.use_pipeline:
            # Adaptive pipeline: dynamically adjusts batch size per worker
            embeddings, details = await distribute_with_adaptive_pipeline(
                scheduler=self.scheduler,
                inputs=inputs,
                process_func=process_on_instance,
                enable_perf_tracking=self.enable_perf_tracking,
                perf_tracker=self.perf_tracker,
                min_batch_size=MIN_BATCH_SIZE,
                max_batch_size=MAX_BATCH_SIZE,
                probe_batch_size=self.micro_batch_size,
            )
        else:
            embeddings, details = await distribute_with_scheduler(
                scheduler=self.scheduler,
                inputs=inputs,
                process_func=process_on_instance,
                enable_perf_tracking=self.enable_perf_tracking,
                perf_tracker=self.perf_tracker,
            )

        # Print performance analysis periodically
        if self.enable_perf_tracking:
            self._request_counter += 1

        return embeddings

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

    async def _process_lsh_batch(
        self,
        all_inputs: list[str],
        healthy: list,
        normalize: bool,
        truncate: bool,
        bitn: int,
    ) -> list[list[float]]:
        """Process a batch of inputs for LSH.

        Returns embeddings (to be converted to LSH hashes by caller).
        """
        import time as _time

        t1 = _time.perf_counter()

        embeddings = await self._distribute_with_scheduler(
            all_inputs, healthy, normalize, truncate
        )

        t2 = _time.perf_counter()

        if self.enable_perf_tracking:
            logger.mesg(
                f"[Batch embed] n={len(all_inputs)}, time={((t2-t1)*1000):.1f}ms"
            )

        return embeddings

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

                embeddings = await self._distribute_with_scheduler(
                    inputs, healthy, request.normalize, request.truncate
                )

                t2 = _time.perf_counter()

            # Convert to numpy array for vectorized LSH (outside lock)
            embs_array = np.array(embeddings, dtype=np.float32)
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
                array_conv = (t3 - t2) * 1000
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
                    f"embed={embed_time:.1f}ms, np.array={array_conv:.1f}ms, "
                    f"lsh={lsh_time:.1f}ms{gap_info}"
                )

            self._pending_requests -= 1
            return lsh_hashes

        except Exception as e:
            self._pending_requests -= 1
            self.stats.total_errors += 1
            logger.warn(f"× LSH error: {e}")
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


class TEIMachineArgParser:
    """Argument parser for TEI Machine CLI."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="TEI Machine - Load-balanced proxy for TEI instances",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        """Setup all command-line arguments."""
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
            help="Comma-separated list of TEI endpoints (skip auto-discovery)",
        )
        self.parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=BATCH_SIZE,
            help=f"Max batch size per instance (default: {BATCH_SIZE})",
        )
        self.parser.add_argument(
            "-m",
            "--micro-batch-size",
            type=int,
            default=MICRO_BATCH_SIZE,
            help=f"Micro-batch size for pipeline scheduling (default: {MICRO_BATCH_SIZE})",
        )
        self.parser.add_argument(
            "-t",
            "--timeout",
            type=float,
            default=60.0,
            help="Request timeout in seconds (default: 60)",
        )
        self.parser.add_argument(
            "--no-gpu-lsh",
            action="store_true",
            help="Disable GPU acceleration for LSH computation (use CPU instead)",
        )
        self.parser.add_argument(
            "--perf-track",
            action="store_true",
            help="Enable detailed performance tracking to identify bottlenecks",
        )
        self.parser.add_argument(
            "--no-pipeline",
            action="store_true",
            help="Disable pipeline scheduling (use round-based scheduling instead)",
        )
        self.parser.add_argument(
            "action",
            nargs="?",
            choices=["run", "discover", "health"],
            default=None,
            help="Action to perform: run (start server), discover (list instances), health (check health)",
        )


def discover_instances(args) -> list[TEIInstance]:
    """Discover or create TEI instances based on args."""
    if args.endpoints:
        endpoints = [e.strip() for e in args.endpoints.split(",")]
        instances = TEIInstanceDiscovery.from_endpoints(endpoints)
        logger.okay(f"[tei_machine] Using {len(instances)} manual endpoints")
    else:
        instances = TEIInstanceDiscovery.discover(args.name_pattern)
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


def main():
    """Main entry point."""
    arg_parser = TEIMachineArgParser()
    args = arg_parser.args

    # Show help if no action specified
    if args.action is None:
        arg_parser.parser.print_help()
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
                "× No TEI instances found. Use -e to specify endpoints manually."
            )
            return

        use_pipeline = not args.no_pipeline

        if args.perf_track:
            logger.note("[tei_machine] Performance tracking ENABLED")
            logger.note(
                "[tei_machine] Detailed metrics will be logged for each request"
            )

        if use_pipeline:
            logger.note(
                f"[tei_machine] Pipeline mode ENABLED (micro_batch_size={args.micro_batch_size})"
            )
        else:
            logger.note("[tei_machine] Round-based scheduling (pipeline disabled)")

        server = TEIMachineServer(
            instances=instances,
            port=args.port,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            timeout=args.timeout,
            use_gpu_lsh=not args.no_gpu_lsh,
            enable_perf_tracking=args.perf_track,
            use_pipeline=use_pipeline,
        )
        server.run()


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_machine.py#machine-clis
