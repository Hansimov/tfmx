"""TEI (Text Embeddings Inference) Machine Manager

This module provides a load-balanced proxy server that distributes embedding
requests across multiple TEI Docker instances running on different GPUs.
"""

# ANCHOR[id=machine-clis]
CLI_EPILOG = """
Examples:
  # Start machine server (auto-discover TEI containers)
  tei_machine run                   # Start on default port 28800
  tei_machine run -p 28800          # Start on specific port
  
  # Filter containers by name pattern
  tei_machine run -n "qwen3-embedding"  # Only match containers with this pattern
  
  # Manual endpoint specification (skip auto-discovery)
  tei_machine run -e "http://localhost:28880,http://localhost:28881"
  
  # With custom batch size per instance
  tei_machine run -b 50             # Max 50 inputs per request to each instance
  
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
import uvicorn

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from itertools import cycle
from pydantic import BaseModel, Field
from tclogger import logger, logstr
from typing import Optional, Union
from webu import setup_swagger_ui


PORT = 28800
BATCH_SIZE = 100
TEI_CONTAINER_IMAGE_PATTERN = "text-embeddings-inference"


# ============================================================================
# Pydantic Models for API
# ============================================================================


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


class InfoResponse(BaseModel):
    """Response model for info endpoint."""

    port: int = Field(..., description="Machine server port")
    instances: list[InstanceInfo] = Field(..., description="List of TEI instances")
    stats: MachineStats = Field(..., description="Machine statistics")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")


# ============================================================================
# Data Classes
# ============================================================================


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

    def to_model(self) -> MachineStats:
        """Convert to Pydantic model."""
        return MachineStats(
            total_requests=self.total_requests,
            total_inputs=self.total_inputs,
            total_errors=self.total_errors,
            requests_per_instance=self.requests_per_instance,
        )


# ============================================================================
# Instance Discovery
# ============================================================================


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


# ============================================================================
# Load Balancer
# ============================================================================


class TEILoadBalancer:
    """Load balancer for distributing requests across TEI instances."""

    def __init__(self, instances: list[TEIInstance]):
        self.instances = instances
        self.healthy_instances: list[TEIInstance] = []
        self._instance_cycle = None
        self._lock = asyncio.Lock()

    def update_healthy_instances(self) -> None:
        """Update the list of healthy instances and reset the cycle."""
        self.healthy_instances = [i for i in self.instances if i.healthy]
        if self.healthy_instances:
            self._instance_cycle = cycle(self.healthy_instances)
        else:
            self._instance_cycle = None

    async def get_next_instance(self) -> Optional[TEIInstance]:
        """Get the next healthy instance using round-robin."""
        async with self._lock:
            if not self._instance_cycle:
                return None
            return next(self._instance_cycle)

    def get_all_healthy(self) -> list[TEIInstance]:
        """Get all healthy instances for parallel distribution."""
        return self.healthy_instances.copy()


# ============================================================================
# Machine Server (FastAPI)
# ============================================================================


class TEIMachineServer:
    """FastAPI server that proxies requests to multiple TEI instances."""

    def __init__(
        self,
        instances: list[TEIInstance],
        port: int = PORT,
        batch_size: int = BATCH_SIZE,
        timeout: float = 60.0,
    ):
        self.instances = instances
        self.port = port
        self.batch_size = batch_size
        self.timeout = timeout
        self.load_balancer = TEILoadBalancer(instances)
        self.stats = TEIMachineStatsData()
        self._client: Optional[httpx.AsyncClient] = None
        self._health_task: Optional[asyncio.Task] = None

        # Create FastAPI app
        self.app = self._create_app()

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
        app.post(
            "/embed",
            response_model=list[list[float]],
            summary="Generate embeddings",
            description="Generate embeddings for input texts using load-balanced TEI instances",
        )(self.handle_embed)

        app.get(
            "/health",
            response_model=HealthResponse,
            summary="Health check",
            description="Check health status of the machine",
        )(self.handle_health)

        app.get(
            "/info",
            response_model=InfoResponse,
            summary="Machine info",
            description="Get detailed information about the machine and statistics",
        )(self.handle_info)

        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage application lifecycle."""
        # Startup
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))

        # Initial health check
        await self.health_check_all()

        if not self.load_balancer.healthy_instances:
            logger.warn("× No healthy TEI instances available at startup")

        # Start background health checker
        self._health_task = asyncio.create_task(self._periodic_health_check())

        logger.okay(f"[tei_machine] Started on port {self.port}")
        logger.mesg(
            f"[tei_machine] Healthy instances: "
            f"{len(self.load_balancer.healthy_instances)}/{len(self.instances)}"
        )

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
        self.load_balancer.update_healthy_instances()

    async def _check_instance_health(self, instance: TEIInstance) -> bool:
        """Check health of a single instance."""
        try:
            resp = await self._client.get(instance.health_url)
            instance.healthy = resp.status_code == 200
            return instance.healthy
        except Exception:
            instance.healthy = False
            return False

    async def handle_embed(self, request: EmbedRequest) -> list[list[float]]:
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
        healthy = self.load_balancer.get_all_healthy()
        if not healthy:
            self.stats.total_errors += 1
            raise HTTPException(
                status_code=503, detail="No healthy instances available"
            )

        try:
            # Distribute inputs across instances
            embeddings = await self._distribute_and_collect(
                inputs, healthy, request.normalize, request.truncate
            )
            return embeddings

        except Exception as e:
            self.stats.total_errors += 1
            logger.warn(f"× Embed error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _distribute_and_collect(
        self,
        inputs: list[str],
        instances: list[TEIInstance],
        normalize: bool,
        truncate: bool,
    ) -> list[list[float]]:
        """
        Distribute inputs across instances and collect results.

        Strategy: Split inputs evenly across all healthy instances for maximum parallelism.
        """
        n_instances = len(instances)
        n_inputs = len(inputs)

        # Calculate chunk size per instance
        chunk_size = max(1, (n_inputs + n_instances - 1) // n_instances)

        # Create tasks for each chunk
        tasks = []
        chunk_indices = []  # Track original indices for ordering

        for i, instance in enumerate(instances):
            start_idx = i * chunk_size
            if start_idx >= n_inputs:
                break
            end_idx = min(start_idx + chunk_size, n_inputs)
            chunk = inputs[start_idx:end_idx]

            if chunk:
                chunk_indices.append((start_idx, end_idx))
                tasks.append(
                    self._send_embed_request(instance, chunk, normalize, truncate)
                )

                # Update stats
                instance_name = instance.container_name
                self.stats.requests_per_instance[instance_name] = (
                    self.stats.requests_per_instance.get(instance_name, 0) + 1
                )

        # Execute all requests in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results in order
        all_embeddings = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warn(f"× Instance error: {result}")
                # Return empty embeddings for failed chunks
                chunk_len = chunk_indices[i][1] - chunk_indices[i][0]
                all_embeddings.extend([[]] * chunk_len)
            else:
                all_embeddings.extend(result)

        return all_embeddings

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

    async def handle_health(self) -> HealthResponse:
        """Handle health check requests."""
        healthy_count = len(self.load_balancer.healthy_instances)
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

    async def handle_info(self) -> InfoResponse:
        """Handle info requests."""
        return InfoResponse(
            port=self.port,
            instances=[inst.to_info() for inst in self.instances],
            stats=self.stats.to_model(),
        )

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


# ============================================================================
# CLI
# ============================================================================


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
            "-t",
            "--timeout",
            type=float,
            default=60.0,
            help="Request timeout in seconds (default: 60)",
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

        server = TEIMachineServer(
            instances=instances,
            port=args.port,
            batch_size=args.batch_size,
            timeout=args.timeout,
        )
        server.run()


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_machine.py#machine-clis
