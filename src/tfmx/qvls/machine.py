"""QVL (Qwen3-VL) Machine Manager

A load-balanced proxy server that distributes chat completion requests
across multiple vLLM Docker instances running on different GPUs.
"""

# ANCHOR[id=qvl-machine-clis]
CLI_EPILOG = """
Examples:
  # Start machine server (auto-discover vLLM containers)
  qvl_machine run                   # Start on default port 29800
  qvl_machine run -p 29800          # Start on specific port

  # Filter containers by name pattern
  qvl_machine run -n "qvl--qwen"    # Only match containers with this pattern

  # Manual endpoint specification (skip auto-discovery)
  qvl_machine run -e "http://localhost:29880,http://localhost:29881"

  # Performance tracking
  qvl_machine run --perf-track      # Enable detailed performance tracking

  # Check discovered instances without starting server
  qvl_machine discover              # List all discovered vLLM instances

  # Health check all instances
  qvl_machine health                # Check health of all instances
"""

import argparse
import asyncio
import re
import subprocess
import time

import httpx
import orjson
import uvicorn

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from tclogger import logger, logstr
from typing import Optional, Union
from webu import setup_swagger_ui

from .compose import MAX_CONCURRENT_REQUESTS, MACHINE_PORT, SERVER_PORT


PORT = MACHINE_PORT
MAX_CONCURRENT = MAX_CONCURRENT_REQUESTS
VLLM_CONTAINER_IMAGE_PATTERN = "vllm"


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion endpoint."""

    model: str = Field(default="", description="Model name")
    messages: list[dict] = Field(
        ...,
        description="Chat messages in OpenAI format",
        examples=[[{"role": "user", "content": "Hello!"}]],
    )
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False, description="Enable streaming")


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


@dataclass
class VLLMInstance:
    """Represents a single vLLM Docker instance."""

    container_name: str
    host: str
    port: int
    gpu_id: Optional[int] = None
    healthy: bool = False
    _active_requests: int = 0

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
        return (
            f"VLLMInstance({status} {self.container_name} "
            f"@ {self.endpoint}, {gpu_info})"
        )

    def to_info(self) -> InstanceInfo:
        return InstanceInfo(
            name=self.container_name,
            endpoint=self.endpoint,
            gpu_id=self.gpu_id,
            healthy=self.healthy,
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

        # Round-robin index for distributing requests
        self._rr_lock: Optional[asyncio.Lock] = None

        self.app = self._create_app()

    def get_healthy_instances(self) -> list[VLLMInstance]:
        return [i for i in self.instances if i.healthy]

    def _get_idle_instance(self) -> Optional[VLLMInstance]:
        """Get the instance with most available slots."""
        idle = [i for i in self.instances if i.healthy and i.is_idle]
        if not idle:
            return None
        idle.sort(key=lambda i: i.available_slots, reverse=True)
        return idle[0]

    def _create_app(self) -> FastAPI:
        app = FastAPI(
            title="QVL Machine",
            description="Load-balanced proxy for Qwen3-VL vLLM instances",
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
            summary="Chat completion",
            description="Distribute chat completion to vLLM instances",
        )(self.chat_completions)

        return app

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        self._rr_lock = asyncio.Lock()

        await self.health_check_all()

        healthy = self.get_healthy_instances()
        if not healthy:
            logger.warn("× No healthy vLLM instances available at startup")

        self._health_task = asyncio.create_task(self._periodic_health_check())

        logger.okay(f"[qvl_machine] Started on port {self.port}")
        healthy_str = logstr.okay(len(healthy))
        total_str = logstr.mesg(len(self.instances))
        logger.mesg(f"[qvl_machine] Healthy instances: {healthy_str}/{total_str}")

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

    async def info(self) -> InfoResponse:
        return InfoResponse(
            port=self.port,
            instances=[inst.to_info() for inst in self.instances],
            stats=self.stats.to_model(),
        )

    async def chat_completions(self, request: Request) -> Response:
        """Forward chat completion request to an idle vLLM instance.

        Selects the least-loaded instance and proxies the request.
        """
        t0 = time.perf_counter()
        self.stats.total_requests += 1
        self.stats.active_requests += 1

        try:
            # Get an idle instance
            instance = self._get_idle_instance()
            if instance is None:
                # Wait briefly for a slot to free up
                for _ in range(10):
                    await asyncio.sleep(0.1)
                    instance = self._get_idle_instance()
                    if instance:
                        break

            if instance is None:
                self.stats.total_errors += 1
                raise HTTPException(
                    status_code=503, detail="No available vLLM instances"
                )

            instance._active_requests += 1

            try:
                # Read raw request body and forward to vLLM
                body = await request.body()

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
                if resp.status_code == 200:
                    try:
                        data = orjson.loads(resp.content)
                        usage = data.get("usage", {})
                        self.stats.total_tokens += usage.get("total_tokens", 0)
                    except Exception:
                        pass

                latency = time.perf_counter() - t0
                if self.enable_perf_tracking:
                    logger.mesg(
                        f"[qvl_machine] {instance_name} "
                        f"status={resp.status_code} latency={latency:.2f}s"
                    )

                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                    media_type="application/json",
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
            "action",
            nargs="?",
            choices=["run", "discover", "health"],
            default=None,
            help="Action: run (start server), discover (list instances), health (check health)",
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
                f"{gpu_str:<6} {inst.container_name:<40} "
                f"{inst.endpoint:<25} {status:<8}"
            )
        else:
            logger.mesg(f"{gpu_str:<6} {inst.container_name:<45} {inst.endpoint:<25}")

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
    """Main entry point."""
    arg_parser = QVLMachineArgParser()
    args = arg_parser.args

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
                "× No vLLM instances found. " "Use -e to specify endpoints manually."
            )
            return

        server = QVLMachineServer(
            instances=instances,
            port=args.port,
            timeout=args.timeout,
            enable_perf_tracking=args.perf_track,
        )
        server.run()


if __name__ == "__main__":
    main()
