"""TEI (Text Embeddings Inference) Docker Compose Manager"""

# ANCHOR[id=clis]
CLI_EPILOG = """
Examples:
  # Set model as environment variable for convenience
  # export MODEL="Alibaba-NLP/gte-multilingual-base"
  export MODEL="Qwen/Qwen3-Embedding-0.6B"
  
  # Basic operations
  tei_compose up                    # Start on all GPUs (auto fallback on failure)
  tei_compose ps                    # Check container status
  tei_compose logs                  # View recent logs
  tei_compose stop                  # Stop containers (keep them)
  tei_compose start                 # Start stopped containers
  tei_compose restart               # Restart containers
  tei_compose down                  # Stop and remove containers
  tei_compose generate              # Generate compose file only
  tei_compose health                # Check GPU health status
  
  # With specific model
  tei_compose -m "$MODEL" generate  # Generate compose file only
  tei_compose -m "$MODEL" up        # Start with specified model
  tei_compose -m "Alibaba-NLP/gte-multilingual-base" up  # Use model name directly
  
  # With specific GPUs
  tei_compose -g "0,1" up           # Start on GPU 0 and 1
  tei_compose -g "2" up             # Start on GPU 2 only
  
  # Custom port and project name
  tei_compose -p 28890 up           # Use port 28890 as base
  tei_compose -j my-tei up          # Custom project name
  
  # With HuggingFace token for private models
  tei_compose -t hf_**** up         # Use HF token
  
  # Advanced log viewing
  tei_compose logs -f               # Follow logs in real-time
  tei_compose logs --tail 200       # Show last 200 lines
  tei_compose logs -f --tail 50     # Follow with 50 lines buffer

Startup Strategy:
  The 'up' command uses a smart two-phase startup approach:
  1. First attempts to start all services together (fastest)
  2. If batch start fails (e.g., GPU errors), automatically falls back
     to individual service startup with health checks
  3. Each GPU failure is isolated and won't prevent other GPUs from starting
  
  This provides optimal performance when all GPUs are healthy, while
  maintaining robustness when some GPUs have issues.

Health Check:
  - Use 'tei_compose health' to diagnose GPU issues before deployment
  - Individual startup mode pre-checks each GPU before starting its container
"""

import argparse
import re
import shutil
import subprocess

from pathlib import Path
from typing import Optional

from tclogger import logger


SERVER_PORT = 28880
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
HF_ENDPOINT = "https://hf-mirror.com"
CACHE_HF = ".cache/huggingface"
CACHE_HF_HUB = f"{CACHE_HF}/hub"

# TEI image tag mapping by GPU compute capability
TEI_IMAGE_TAGS = {
    "8.0": "1.8",  # Ampere 80 (A100, A30)
    "8.6": "86-1.8",  # Ampere 86 (A10, A40, RTX 3080)
    "8.9": "89-1.8",  # Ada Lovelace (RTX 4090)
    "9.0": "1.8",  # Hopper (H100)
}
TEI_TAG = "86-1.8"  # Fallback

TEI_IMAGE_BASE = "ghcr.io/huggingface/text-embeddings-inference"
TEI_IMAGE_MIRROR = "m.daocloud.io"

MAX_CLIENT_BATCH_SIZE = 300


class GPUInfo:
    """Information about a single GPU."""

    def __init__(self, index: int, compute_cap: str):
        self.index = index
        self.compute_cap = compute_cap
        self.arch_tag = TEI_IMAGE_TAGS.get(compute_cap, TEI_TAG)
        self.image = f"{TEI_IMAGE_BASE}:{self.arch_tag}"

    def __repr__(self):
        return f"GPU({self.index}, cap={self.compute_cap}, tag={self.arch_tag})"


class GPUDetector:
    """GPU detection and management."""

    @staticmethod
    def detect(gpu_ids: Optional[str] = None) -> list[GPUInfo]:
        """Detect available GPUs and their compute capabilities."""
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(",")
                    index = int(parts[0].strip())
                    compute_cap = parts[1].strip()
                    gpus.append(GPUInfo(index, compute_cap))

            # Filter by specified GPU IDs
            if gpu_ids:
                specified = [int(x.strip()) for x in gpu_ids.split(",")]
                gpus = [g for g in gpus if g.index in specified]

            return gpus
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warn(f"× Failed to detect GPUs: {e}")
            return []

    @staticmethod
    def check_gpu_health(gpu_index: int) -> tuple[bool, str]:
        """Check if a specific GPU is healthy and accessible.

        Returns:
            (is_healthy, message) tuple
        """
        try:
            # Query specific GPU for basic info
            result = subprocess.run(
                f"nvidia-smi -i {gpu_index} --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Unknown error"
                return False, f"nvidia-smi failed: {error_msg}"

            # Check for NVML errors in output
            if "error" in result.stdout.lower() or "failed" in result.stdout.lower():
                return False, f"GPU error: {result.stdout.strip()}"

            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "nvidia-smi timeout"
        except Exception as e:
            return False, f"Health check failed: {e}"


class ModelConfigManager:
    """Manages HuggingFace model configuration files."""

    def __init__(self, cache_hf_hub: str = CACHE_HF_HUB):
        self.cache_hf_hub = cache_hf_hub

    @staticmethod
    def get_tfmx_src_dir() -> Path:
        """Get the tfmx source directory."""
        # First check if running from source
        src_dir = Path(__file__).resolve().parent
        if (src_dir / "config_sentence_transformers.json").exists():
            return src_dir

        # Fallback to home repos
        home_src = Path.home() / "repos" / "tfmx" / "src" / "tfmx"
        if home_src.exists():
            return home_src

        return src_dir

    def get_model_snapshot_dir(self, model_name: str) -> Optional[Path]:
        """Find the model snapshot directory in HuggingFace cache."""
        model_name_dash = model_name.replace("/", "--")
        cache_path = Path.home() / self.cache_hf_hub

        if not cache_path.exists():
            return None

        # Find snapshot directory
        model_dir = cache_path / f"models--{model_name_dash}"
        if not model_dir.exists():
            return None

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        # Get the first snapshot (usually there's only one)
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                return snapshot

        return None

    def patch_config_files(self, model_name: str) -> None:
        """Patch config files to fix issues with some models."""
        snapshot_dir = self.get_model_snapshot_dir(model_name)
        if not snapshot_dir:
            logger.mesg(f"[tfmx] Model cache not found, skipping patch")
            return

        tfmx_src = self.get_tfmx_src_dir()

        # Patch config_sentence_transformers.json
        self._patch_sentence_transformers_config(snapshot_dir, tfmx_src)

        # Patch config.json (check for corruption)
        self._patch_main_config(snapshot_dir, tfmx_src)

    def _patch_sentence_transformers_config(
        self, snapshot_dir: Path, tfmx_src: Path
    ) -> None:
        """Patch config_sentence_transformers.json if missing."""
        config_name = "config_sentence_transformers.json"
        target = snapshot_dir / config_name
        source = tfmx_src / config_name

        if target.exists():
            logger.mesg(f"[tfmx] Skip existed: '{target}'")
        elif source.exists():
            shutil.copy(source, target)
            logger.okay(f"[tfmx] Copied: '{target}'")

    def _patch_main_config(self, snapshot_dir: Path, tfmx_src: Path) -> None:
        """Patch config.json, fixing corruption if needed."""
        config_name = "config.json"
        target = snapshot_dir / config_name
        source = tfmx_src / "config_qwen3_embedding_06b.json"

        if target.exists():
            # Check if file is corrupted (doesn't end with })
            content = target.read_text().strip()
            if not content.endswith("}"):
                logger.warn(f"[tfmx] Corrupted: '{target}'")
                target.unlink()
                if source.exists():
                    shutil.copy(source, target)
                    logger.okay(f"[tfmx] Patched: '{target}'")
            else:
                logger.mesg(f"[tfmx] Skip existed: '{target}'")
        elif source.exists():
            shutil.copy(source, target)
            logger.okay(f"[tfmx] Copied: '{target}'")


class DockerImageManager:
    """Manages Docker image operations."""

    @staticmethod
    def ensure_image(image: str) -> bool:
        """Ensure TEI image is available, pull from mirror if needed."""
        # Check if image exists locally
        result = subprocess.run(
            f"docker image inspect {image}",
            shell=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return True

        # Pull from mirror
        mirror_image = f"{TEI_IMAGE_MIRROR}/{image}"
        logger.mesg(f"[tfmx] Pulling image from mirror: {mirror_image}")

        try:
            subprocess.run(
                f"docker pull {mirror_image}",
                shell=True,
                check=True,
            )
            subprocess.run(
                f"docker tag {mirror_image} {image}",
                shell=True,
                check=True,
            )
            logger.okay(f"[tfmx] Image tagged as: {image}")
            return True
        except subprocess.CalledProcessError as e:
            logger.warn(f"× Failed to pull image: {e}")
            return False


class ComposeFileGenerator:
    """Generates docker-compose.yml content for TEI deployment."""

    def __init__(
        self,
        gpus: list[GPUInfo],
        model_name: str,
        port: int,
        project_name: str,
        data_dir: Path,
        hf_token: Optional[str] = None,
        cache_hf: str = CACHE_HF,
        cache_hf_hub: str = CACHE_HF_HUB,
        hf_endpoint: str = HF_ENDPOINT,
    ):
        self.gpus = gpus
        self.model_name = model_name
        self.port = port
        self.project_name = project_name
        self.data_dir = data_dir
        self.hf_token = hf_token
        self.cache_hf = cache_hf
        self.cache_hf_hub = cache_hf_hub
        self.hf_endpoint = hf_endpoint

    def generate(self) -> str:
        """Generate docker-compose.yml content with YAML anchors for common config."""
        lines = self._generate_header()
        # Add x-common-config anchor for shared configuration
        lines.extend(self._generate_common_config())
        lines.append("services:")
        for i, gpu in enumerate(self.gpus):
            service_lines = self._generate_service(gpu=gpu)
            lines.extend(service_lines)
        return "\n".join(lines)

    def _generate_header(self) -> list[str]:
        """Generate compose file header."""
        return [
            f"# TEI Multi-GPU Deployment",
            f"# Model: {self.model_name}",
            f"# GPUs: {[g.index for g in self.gpus]}",
            f"",
            f"name: {self.project_name}",
            f"",
        ]

    def _generate_common_config(self) -> list[str]:
        """Generate common configuration as YAML anchor."""
        lines = [
            "x-common-config: &common-config",
            f"  volumes:",
            f"    - ${{HOME}}/{self.cache_hf}:/root/{self.cache_hf}",
            f"    - {self.data_dir}:/data",
            f"  environment:",
            f"    - HF_ENDPOINT={self.hf_endpoint}",
            f"    - HF_HOME=/root/{self.cache_hf}",
            f"    - HF_HUB_CACHE=/root/{self.cache_hf_hub}",
            f"    - HUGGINGFACE_HUB_CACHE=/root/{self.cache_hf_hub}",
            f"  command:",
            f"    - --huggingface-hub-cache",
            f"    - /root/{self.cache_hf_hub}",
            f"    - --model-id",
            f"    - {self.model_name}",
        ]
        if self.hf_token:
            lines.extend([f"    - --hf-token", f"    - {self.hf_token}"])
        lines.extend(
            [
                f"    - --dtype",
                f"    - float16",
                f"    - --max-batch-tokens",
                f'    - "32768"',
                f"    - --max-client-batch-size",
                f'    - "{MAX_CLIENT_BATCH_SIZE}"',
                # f"  restart: unless-stopped",
                f"  healthcheck:",
                f'    test: ["CMD", "curl", "-f", "http://localhost:80/health"]',
                f"    interval: 30s",
                f"    timeout: 10s",
                f"    retries: 3",
                f"    start_period: 60s",
                f"",
            ]
        )
        return lines

    def _generate_service(self, gpu: GPUInfo) -> list[str]:
        """Generate service definition for a single GPU using YAML anchor."""
        service_port = self.port + gpu.index
        container_name = f"{self.project_name}--gpu{gpu.index}"
        lines = [
            f"  tei-gpu{gpu.index}:",
            f"    <<: *common-config",
            f"    image: {gpu.image}",
            f"    container_name: {container_name}",
            f"    ports:",
            f'      - "{service_port}:80"',
            f"    deploy:",
            f"      resources:",
            f"        reservations:",
            f"          devices:",
            f"            - driver: nvidia",
            f'              device_ids: ["{gpu.index}"]',
            f"              capabilities: [gpu]",
            f"",
        ]
        return lines


class TEIComposer:
    """Composer for TEI Docker Compose deployments."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        port: int = SERVER_PORT,
        project_name: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        hf_token: Optional[str] = None,
        compose_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.port = port
        self.gpu_ids = gpu_ids
        self.hf_token = hf_token

        # project name must match: '^[a-z0-9][a-z0-9_-]*$'
        project_dash = model_name.replace("/", "--").lower()
        project_dash = re.sub(r"[^a-z0-9_-]", "_", project_dash)
        self.project_name = project_name or f"tei--{project_dash}"

        # Compose file location (default to script directory)
        if compose_dir:
            self.compose_dir = Path(compose_dir)
        else:
            script_dir = Path(__file__).resolve().parent
            self.compose_dir = script_dir / "compose"

        self.compose_file = self.compose_dir / f"{self.project_name}.yml"

        # Components
        self.gpus = GPUDetector.detect(gpu_ids)
        self.model_config_manager = ModelConfigManager()
        self.image_manager = DockerImageManager()

    def _get_service_name(self, gpu: GPUInfo) -> str:
        """Get the docker compose service name for a GPU."""
        return f"tei-gpu{gpu.index}"

    def _get_container_name(self, gpu: GPUInfo) -> str:
        """Get the container name for a GPU."""
        return f"{self.project_name}--gpu{gpu.index}"

    def _ensure_compose_dir(self) -> None:
        """Ensure compose directory exists."""
        self.compose_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_data_dir(self) -> Path:
        """Ensure docker_data directory exists."""
        script_dir = Path(__file__).resolve().parent
        data_dir = script_dir / "docker_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def generate_compose_file(self) -> Path:
        """Generate the docker-compose.yml file."""
        self._ensure_compose_dir()
        data_dir = self._ensure_data_dir()
        compose_generator = ComposeFileGenerator(
            gpus=self.gpus,
            model_name=self.model_name,
            port=self.port,
            project_name=self.project_name,
            data_dir=data_dir,
            hf_token=self.hf_token,
        )
        content = compose_generator.generate()
        self.compose_file.write_text(content)
        logger.okay(f"[tfmx] Generated: {self.compose_file}")
        return self.compose_file

    def _run_compose_cmd(
        self, cmd: str, capture_output: bool = False
    ) -> subprocess.CompletedProcess:
        """Run a docker compose command."""
        full_cmd = f"docker compose -f {self.compose_file} {cmd}"
        logger.mesg(f"[tfmx] Running: {full_cmd}")
        return subprocess.run(full_cmd, shell=True, capture_output=capture_output)

    def _run_service_cmd(
        self, cmd: str, gpu: GPUInfo, capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a docker compose command for a specific service."""
        service_name = self._get_service_name(gpu)
        full_cmd = f"docker compose -f {self.compose_file} {cmd} {service_name}"
        return subprocess.run(
            full_cmd, shell=True, capture_output=capture_output, text=True
        )

    def _start_service_isolated(
        self, gpu: GPUInfo, check_health: bool = True
    ) -> tuple[bool, str]:
        """Start a single service in isolation with error handling.

        Args:
            gpu: GPU info for the service to start
            check_health: Whether to pre-check GPU health

        Returns:
            (success, message) tuple
        """
        service_name = self._get_service_name(gpu)
        container_name = self._get_container_name(gpu)

        # Pre-check GPU health
        if check_health:
            is_healthy, health_msg = GPUDetector.check_gpu_health(gpu.index)
            if not is_healthy:
                return False, f"GPU {gpu.index} unhealthy: {health_msg}"

        # Try to start the service
        try:
            result = self._run_service_cmd("up -d", gpu, capture_output=True)
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                # Check for common NVML errors
                if "nvml" in error_msg.lower() or "nvidia" in error_msg.lower():
                    return False, f"NVIDIA runtime error: {error_msg}"
                return False, f"Failed to start: {error_msg}"

            # Verify container is running
            check_result = subprocess.run(
                f'docker ps --filter "name=^/{container_name}$" --format "{{{{.Status}}}}"',
                shell=True,
                capture_output=True,
                text=True,
            )
            status = check_result.stdout.strip()
            if status and status.startswith("Up"):
                return True, f"Started successfully"
            else:
                return (
                    False,
                    f"Container not running after start: {status or 'not found'}",
                )

        except Exception as e:
            return False, f"Exception during start: {e}"

    def _start_all_together(self) -> tuple[bool, str]:
        """Try to start all services together.

        Returns:
            (success, message) tuple
        """
        try:
            result = self._run_compose_cmd("up -d", capture_output=True)
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return False, error_msg
            return True, "All services started together"
        except Exception as e:
            return False, f"Exception: {e}"

    def _start_individually(self) -> dict[int, tuple[bool, str]]:
        """Start each GPU service independently with error handling.

        Returns:
            Dictionary mapping GPU index to (success, message) tuples
        """
        results: dict[int, tuple[bool, str]] = {}
        logger.mesg(f"[tfmx] Starting services individually...")

        for gpu in self.gpus:
            service_name = self._get_service_name(gpu)
            logger.mesg(f"[tfmx] Starting {service_name} (GPU {gpu.index})...")
            success, message = self._start_service_isolated(gpu, check_health=True)
            results[gpu.index] = (success, message)

            if success:
                logger.okay(f"  ✓ GPU {gpu.index}: {message}")
            else:
                logger.warn(f"  × GPU {gpu.index}: {message}")

        return results

    def up(self) -> None:
        """Start all TEI containers.

        Strategy:
        1. Try to start all services together for efficiency
        2. If batch start fails, fall back to individual startup with error isolation
        """
        if not self.gpus:
            logger.warn("× No GPUs detected")
            return

        logger.mesg(f"[tfmx] Starting TEI for model: {self.model_name}")
        logger.mesg(
            f"[tfmx] GPUs: {[f'{g.index}(cap={g.compute_cap})' for g in self.gpus]}"
        )

        # Patch config files
        self.model_config_manager.patch_config_files(self.model_name)

        # Ensure images are available
        images = set(g.image for g in self.gpus)
        for image in images:
            self.image_manager.ensure_image(image)

        # Ensure directories
        self._ensure_data_dir()

        # Generate compose file
        self.generate_compose_file()

        # Try to start all services together first
        logger.mesg(f"[tfmx] Attempting to start all services together...")
        success, message = self._start_all_together()

        if success:
            logger.okay(f"[tfmx] {message}")
            # Verify all containers are running
            all_running = True
            for gpu in self.gpus:
                container_name = self._get_container_name(gpu)
                check_result = subprocess.run(
                    f'docker ps --filter "name=^/{container_name}$" --format "{{{{.Status}}}}"',
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                status = check_result.stdout.strip()
                if not status or not status.startswith("Up"):
                    all_running = False
                    break

            if all_running:
                logger.okay(
                    f"[tfmx] All GPU services started successfully: {[g.index for g in self.gpus]}"
                )
            else:
                logger.warn(
                    f"[tfmx] Some containers failed to start, falling back to individual startup..."
                )
                # Clean up partial deployment
                self._run_compose_cmd("down", capture_output=True)
                # Start individually
                results = self._start_individually()
                self._show_startup_summary(results)
        else:
            logger.warn(f"[tfmx] Batch startup failed: {message}")
            logger.mesg(f"[tfmx] Falling back to individual service startup...")
            # Start individually
            results = self._start_individually()
            self._show_startup_summary(results)

        # Show status
        self.ps()

    def _show_startup_summary(self, results: dict[int, tuple[bool, str]]) -> None:
        """Show summary of individual startup results."""
        succeeded = [idx for idx, (ok, _) in results.items() if ok]
        failed = [idx for idx, (ok, _) in results.items() if not ok]

        print()
        if succeeded:
            logger.okay(f"[tfmx] Successfully started: GPU {succeeded}")
        if failed:
            logger.warn(f"[tfmx] Failed to start: GPU {failed}")
            for idx in failed:
                _, msg = results[idx]
                logger.warn(f"  GPU {idx}: {msg}")

    def down(self) -> None:
        """Stop and remove all TEI containers."""
        if not self.compose_file.exists():
            logger.warn(f"× Compose file not found: {self.compose_file}")
            return
        self._run_compose_cmd("down")

    def stop(self) -> None:
        """Stop all TEI containers (keep containers)."""
        if not self.compose_file.exists():
            logger.warn(f"× Compose file not found: {self.compose_file}")
            return
        self._run_compose_cmd("stop")

    def start(self) -> None:
        """Start existing TEI containers."""
        if not self.compose_file.exists():
            logger.warn(f"× Compose file not found: {self.compose_file}")
            return
        self._run_compose_cmd("start")

    def restart(self) -> None:
        """Restart all TEI containers."""
        if not self.compose_file.exists():
            logger.warn(f"× Compose file not found: {self.compose_file}")
            return
        self._run_compose_cmd("restart")

    def ps(self) -> None:
        """Show status of all TEI containers."""
        if not self.compose_file.exists():
            self._show_manual_status()
            return
        self._run_compose_cmd("ps")

    def logs(self, follow: bool = False, tail: int = 100) -> None:
        """Show logs of all TEI containers."""
        if not self.compose_file.exists():
            logger.warn(f"× Compose file not found: {self.compose_file}")
            return
        follow_flag = "-f" if follow else ""
        self._run_compose_cmd(f"logs --tail={tail} {follow_flag}".strip())

    def _show_manual_status(self) -> None:
        """Show status by querying Docker directly."""
        logger.mesg(f"[tfmx] TEI status for: {self.model_name}")
        print("=" * 85)
        print(f"{'GPU':<5} {'CONTAINER':<42} {'PORT':<7} {'CONTAINER':<12} {'GPU':<10}")
        print(f"{'':<5} {'':<42} {'':<7} {'STATUS':<12} {'HEALTH':<10}")
        print("-" * 85)

        for gpu in self.gpus:
            container_name = self._get_container_name(gpu)
            container_port = self.port + gpu.index

            # Check container status
            result = subprocess.run(
                f'docker ps -a --filter "name=^/{container_name}$" --format "{{{{.Status}}}}"',
                shell=True,
                capture_output=True,
                text=True,
            )
            container_status = result.stdout.strip() or "not found"
            if container_status.startswith("Up"):
                container_status = "running"
            elif container_status.startswith("Exited"):
                container_status = "stopped"

            # Check GPU health
            is_healthy, _ = GPUDetector.check_gpu_health(gpu.index)
            gpu_health = "healthy" if is_healthy else "unhealthy"

            print(
                f"{gpu.index:<5} {container_name:<42} {container_port:<7} {container_status:<12} {gpu_health:<10}"
            )

        print("=" * 85)

    def health(self) -> None:
        """Check GPU health status for all GPUs."""
        logger.mesg(f"[tfmx] GPU Health Check")
        print("=" * 70)
        print(f"{'GPU':<6} {'STATUS':<12} {'INFO':<50}")
        print("-" * 70)

        for gpu in self.gpus:
            is_healthy, message = GPUDetector.check_gpu_health(gpu.index)
            status = "healthy" if is_healthy else "unhealthy"
            # Truncate message if too long
            info = message[:48] + ".." if len(message) > 50 else message
            print(f"{gpu.index:<6} {status:<12} {info:<50}")

        print("=" * 70)


class TEIComposeArgParser:
    """Argument parser for TEI Compose CLI."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="TEI Docker Compose Manager",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        """Setup all command-line arguments."""
        # Model configuration
        self.parser.add_argument(
            "-m",
            "--model-name",
            type=str,
            default=MODEL_NAME,
            help=f"Model name (default: {MODEL_NAME})",
        )
        self.parser.add_argument(
            "-p",
            "--port",
            type=int,
            default=SERVER_PORT,
            help=f"Starting port (default: {SERVER_PORT})",
        )
        self.parser.add_argument(
            "-j",
            "--project-name",
            type=str,
            default=None,
            help="Project name (default: tei--MODEL_NAME)",
        )
        self.parser.add_argument(
            "-g",
            "--gpus",
            type=str,
            default=None,
            help="Comma-separated GPU IDs (default: all)",
        )
        self.parser.add_argument(
            "-t",
            "--hf-token",
            type=str,
            default=None,
            help="HuggingFace token for private models",
        )

        # Actions
        self.parser.add_argument(
            "action",
            nargs="?",
            choices=[
                "up",
                "down",
                "stop",
                "start",
                "restart",
                "ps",
                "logs",
                "generate",
                "health",
            ],
            help="Action to perform",
        )
        self.parser.add_argument(
            "-f",
            "--follow",
            action="store_true",
            help="Follow logs (for 'logs' action)",
        )
        self.parser.add_argument(
            "--tail",
            type=int,
            default=100,
            help="Number of log lines to show (default: 100)",
        )


def main():
    arg_parser = TEIComposeArgParser()
    args = arg_parser.args

    # Show help if no action specified
    if not args.action:
        arg_parser.parser.print_help()
        return

    # Use default MODEL_NAME if model_name is empty or whitespace
    model_name = args.model_name.strip() if args.model_name else MODEL_NAME
    if not model_name:
        model_name = MODEL_NAME

    composer = TEIComposer(
        model_name=model_name,
        port=args.port,
        project_name=args.project_name,
        gpu_ids=args.gpus,
        hf_token=args.hf_token,
    )

    if args.action == "up":
        composer.up()
    elif args.action == "down":
        composer.down()
    elif args.action == "stop":
        composer.stop()
    elif args.action == "start":
        composer.start()
    elif args.action == "restart":
        composer.restart()
    elif args.action == "ps":
        composer.ps()
    elif args.action == "logs":
        composer.logs(follow=args.follow, tail=args.tail)
    elif args.action == "generate":
        composer.generate_compose_file()
    elif args.action == "health":
        composer.health()


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_compose.py#clis
