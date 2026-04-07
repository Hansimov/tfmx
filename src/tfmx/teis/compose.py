"""TEI (Text Embeddings Inference) Docker Compose Manager"""

# ANCHOR[id=clis]
CLI_EPILOG = """
Examples:
  # Set model as environment variable for convenience
  # export MODEL="Alibaba-NLP/gte-multilingual-base"
  export MODEL="Qwen/Qwen3-Embedding-0.6B"
    export TEI_PROXY_URL="http://$PROXY_HOST:$PROXY_PORT"
  
  # Basic operations
    tei compose up                    # Start on all healthy GPUs
    tei compose ps                    # Check container status
    tei compose logs                  # View recent logs
    tei compose stop                  # Stop containers (keep them)
    tei compose start                 # Start stopped containers
    tei compose restart               # Restart containers
    tei compose down                  # Stop and remove containers
    tei compose generate              # Generate compose file only
    tei compose health                # Check GPU health status
    tei compose setup                 # Setup model cache (run once before first 'up')
  
  # With specific model
    tei compose generate -m "$MODEL"  # Generate compose file only
    tei compose up -m "$MODEL"        # Start with specified model
    tei compose up -m "Alibaba-NLP/gte-multilingual-base"  # Use model name directly
  
  # With specific GPUs
    tei compose up -g "0,1"           # Start on GPU 0 and 1
    tei compose up -g "2"             # Start on GPU 2 only
        tei compose up --gpu-configs "0,1"  # Per-GPU config with default model
  
  # Custom port and project name
    tei compose up -p 28890           # Use port 28890 as base
    tei compose up -j my-tei          # Custom project name
  
  # With HuggingFace token for private models
    tei compose up -t hf_****         # Use HF token
  
  # With HTTP proxy for downloading models (useful in restricted network environments)
    tei compose up --proxy "$TEI_PROXY_URL"
  
  # Advanced: Manual device mount mode (bypasses nvidia-container-cli)
    tei compose up --mount-mode manual  # Use when nvidia-container-runtime fails due to GPU issues
  
  # Combined: Manual mode with proxy (recommended for GPU issues + network restrictions)
    tei compose up --mount-mode manual --proxy "$TEI_PROXY_URL"
  
  # Advanced log viewing
    tei compose logs -f               # Follow logs in real-time
    tei compose logs --tail 200       # Show last 200 lines
    tei compose logs -f --tail 50     # Follow with 50 lines buffer

Device Mount Modes:
  nvidia-runtime: (default) Uses Docker GPU reservation via nvidia-container-runtime
                  Fast and standard, but may fail if any GPU has NVML issues
  
  manual:         Manually mounts /dev/nvidia* device nodes and driver libraries
                  Bypasses nvidia-container-cli NVML detection, useful when some GPUs
                  have dropped or become inaccessible (e.g., "Unknown Error")
                  Recommended when nvidia-container-runtime fails with NVML errors

HTTP Proxy:
  Use --proxy to set HTTP/HTTPS proxy for model downloads from HuggingFace Hub
  Useful in restricted network environments or when direct access is limited
    Example: --proxy "$TEI_PROXY_URL"

Startup Strategy:
  The 'up' command detects healthy GPUs and starts all services together.
  Pre-checks GPU health to filter out unhealthy GPUs before deployment.

Setup (First Time):
    Run 'tei compose setup' once before first 'up' to pre-create config files.
  This creates empty sentence_*_config.json files in the model cache to skip
  slow downloads from HuggingFace Hub. Only needed once per model.
  
  Example workflow:
    1. huggingface-cli download Qwen/Qwen3-Embedding-0.6B  # Download model
        2. tei compose setup                                    # Create config files
        3. tei compose up                                       # Start containers

Health Check:
    - Use 'tei compose health' to diagnose GPU issues before deployment
  - Unhealthy GPUs are automatically excluded from deployment
"""

import argparse
import re
import shutil
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tclogger import logger

from ..utils.service_bootstrap import wait_for_healthy_http_endpoints


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

# Device mount mode: 'nvidia-runtime' uses docker GPU reservation, 'manual' uses explicit device mounts
# Manual mode bypasses nvidia-container-cli NVML detection, useful when some GPUs have issues
DEVICE_MOUNT_MODE = "manual"  # or "nvidia-runtime"

# Sentence transformer config files that TEI tries to download
# These files are usually empty {} but download is slow due to network restrictions
# We pre-create them in the model cache directory to skip downloads
SENTENCE_CONFIG_FILES = [
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
]


@dataclass
class GpuModelConfig:
    gpu_id: int
    model_name: str = MODEL_NAME

    def __post_init__(self) -> None:
        self.model_name = (self.model_name or MODEL_NAME).strip() or MODEL_NAME

    @property
    def label(self) -> str:
        return self.model_name


def parse_gpu_configs(config_str: str) -> list[GpuModelConfig]:
    configs: list[GpuModelConfig] = []
    seen_gpu_ids: set[int] = set()
    for raw_part in config_str.split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = [field.strip() for field in part.split(":")]
        if not fields or not fields[0]:
            raise ValueError(f"Invalid config: '{part}'. Format: GPU_ID[:MODEL]")
        if len(fields) > 2:
            raise ValueError(f"Invalid config: '{part}'. Format: GPU_ID[:MODEL]")

        gpu_id = int(fields[0])
        if gpu_id in seen_gpu_ids:
            raise ValueError(f"Duplicate GPU ID in config: '{gpu_id}'")
        seen_gpu_ids.add(gpu_id)

        model_name = fields[1] if len(fields) > 1 and fields[1] else MODEL_NAME
        configs.append(GpuModelConfig(gpu_id=gpu_id, model_name=model_name))
    return configs


def infer_gpu_ids(
    gpu_ids: str | None,
    gpu_configs: list[GpuModelConfig] | None = None,
) -> str | None:
    if gpu_ids:
        return gpu_ids
    if not gpu_configs:
        return None

    inferred_ids: list[str] = []
    seen_gpu_ids: set[int] = set()
    for config in gpu_configs:
        if config.gpu_id in seen_gpu_ids:
            continue
        seen_gpu_ids.add(config.gpu_id)
        inferred_ids.append(str(config.gpu_id))
    return ",".join(inferred_ids) if inferred_ids else None


class NvidiaDriverLibs:
    """Detect and manage NVIDIA driver library paths."""

    @staticmethod
    def detect_driver_lib_dir() -> Optional[str]:
        """Detect NVIDIA driver library directory.

        Returns:
            Path to driver libraries, or None if not found
        """
        # Common locations to check
        candidates = [
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/local/nvidia/lib64",
        ]

        for path in candidates:
            if Path(path).exists():
                # Check if libcuda.so exists
                if list(Path(path).glob("libcuda.so*")):
                    return path

        # Fallback: use ldconfig to find libcuda.so
        try:
            result = subprocess.run(
                "ldconfig -p | grep 'libcuda.so\\.' | grep x86-64 | head -1 | awk '{print $NF}'",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                lib_path = result.stdout.strip()
                return str(Path(lib_path).parent)
        except Exception:
            pass

        return None

    @staticmethod
    def get_required_devices() -> list[str]:
        """Get list of required NVIDIA device nodes for manual mounting.

        Returns:
            List of device paths
        """
        return [
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            "/dev/nvidia-modeset",
        ]


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
    def detect(
        gpu_ids: Optional[str] = None, check_health: bool = True
    ) -> list[GPUInfo]:
        """Detect available GPUs and their compute capabilities.

        Args:
            gpu_ids: Comma-separated GPU IDs to filter by
            check_health: Whether to pre-check GPU health and filter unhealthy GPUs

        Returns:
            List of healthy GPUInfo objects
        """
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
            )
            # Note: nvidia-smi may return exit code 0 even with some GPU errors
            # The error messages go to stderr, but stdout still contains healthy GPUs

            gpus = []
            unhealthy_gpus = []

            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            index = int(parts[0].strip())
                            compute_cap = parts[1].strip()
                            gpus.append(GPUInfo(index, compute_cap))
                        except (ValueError, IndexError):
                            continue

            # Check stderr for GPU errors (e.g., dropped GPUs)
            if result.stderr:
                stderr_lower = result.stderr.lower()
                if "error" in stderr_lower or "unable" in stderr_lower:
                    logger.warn(f"[tfmx] nvidia-smi warnings detected:")
                    for line in result.stderr.strip().split("\n"):
                        if line.strip():
                            logger.warn(f"  {line.strip()}")

            # Filter by specified GPU IDs
            if gpu_ids:
                specified = [int(x.strip()) for x in gpu_ids.split(",")]
                gpus = [g for g in gpus if g.index in specified]

            # Health check each GPU if enabled
            if check_health:
                healthy_gpus = []
                for gpu in gpus:
                    is_healthy, msg = GPUDetector.check_gpu_health(gpu.index)
                    if is_healthy:
                        healthy_gpus.append(gpu)
                    else:
                        unhealthy_gpus.append((gpu.index, msg))
                        logger.warn(
                            f"[tfmx] GPU {gpu.index} excluded (unhealthy): {msg}"
                        )
                gpus = healthy_gpus

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

            # Check stderr for errors (nvidia-smi may return 0 even with errors)
            if result.stderr:
                stderr_lower = result.stderr.lower()
                if (
                    "error" in stderr_lower
                    or "unable" in stderr_lower
                    or "unknown" in stderr_lower
                ):
                    return False, f"nvidia-smi error: {result.stderr.strip()}"

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Unknown error"
                return False, f"nvidia-smi failed: {error_msg}"

            # Check for NVML errors in output
            stdout_lower = result.stdout.lower()
            if "error" in stdout_lower or "failed" in stdout_lower:
                return False, f"GPU error: {result.stdout.strip()}"

            # Verify we got valid output (should contain GPU name)
            if not result.stdout.strip():
                return False, "No GPU info returned"

            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "nvidia-smi timeout (GPU may be frozen)"
        except Exception as e:
            return False, f"Health check failed: {e}"

    @staticmethod
    def get_unhealthy_gpu_summary() -> str:
        """Get a summary of any unhealthy GPUs for diagnostic purposes."""
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.stderr and "error" in result.stderr.lower():
                return result.stderr.strip()
            return ""
        except Exception:
            return ""


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
        """Patch config files to fix issues with some models.

        Note: For sentence_*_config.json files, use 'tei compose setup' instead,
        which runs in a Docker container with proper permissions.
        """
        snapshot_dir = self.get_model_snapshot_dir(model_name)
        if not snapshot_dir:
            logger.mesg(f"[tfmx] Model cache not found, skipping patch")
            return

        tfmx_src = self.get_tfmx_src_dir()

        # Patch config.json (check for corruption)
        self._patch_main_config(snapshot_dir, tfmx_src)

    def _patch_main_config(self, snapshot_dir: Path, tfmx_src: Path) -> None:
        """Patch config.json, fixing corruption if needed."""
        config_name = "config.json"
        target = snapshot_dir / config_name
        source = tfmx_src / "config_qwen3_embedding_06b.json"

        if target.exists():
            # Check if file is corrupted (doesn't end with })
            try:
                content = target.read_text().strip()
                if not content.endswith("}"):
                    logger.warn(f"[tfmx] Corrupted: '{target}'")
                    target.unlink()
                    if source.exists():
                        shutil.copy(source, target)
                        logger.okay(f"[tfmx] Patched: '{target}'")
                else:
                    logger.mesg(f"[tfmx] Skip existed: '{target}'")
            except PermissionError:
                logger.mesg(f"[tfmx] Skip (permission): '{target}'")
        elif source.exists():
            try:
                shutil.copy(source, target)
                logger.okay(f"[tfmx] Copied: '{target}'")
            except PermissionError:
                logger.warn(f"[tfmx] Permission denied: '{target}'")


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
        mount_mode: str = "manual",
        driver_lib_dir: Optional[str] = None,
        http_proxy: Optional[str] = None,
        gpu_configs: list[GpuModelConfig] | None = None,
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
        self.mount_mode = mount_mode
        self.driver_lib_dir = driver_lib_dir or NvidiaDriverLibs.detect_driver_lib_dir()
        self.http_proxy = http_proxy
        self.gpu_configs = gpu_configs or []
        self._gpu_config_map = {config.gpu_id: config for config in self.gpu_configs}

    def _get_gpu_config(self, gpu: GPUInfo) -> GpuModelConfig:
        return self._gpu_config_map.get(
            gpu.index,
            GpuModelConfig(gpu_id=gpu.index, model_name=self.model_name),
        )

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
        lines = [f"# TEI Multi-GPU Deployment"]
        if self._gpu_config_map:
            lines.append("# Per-GPU configs:")
            for gpu in self.gpus:
                config = self._get_gpu_config(gpu)
                lines.append(f"#   GPU {config.gpu_id}: {config.label}")
        else:
            lines.append(f"# Model: {self.model_name}")
        lines.extend(
            [
                f"# GPUs: {[g.index for g in self.gpus]}",
                f"",
                f"name: {self.project_name}",
                f"",
            ]
        )
        return lines

    def _generate_common_config(self) -> list[str]:
        """Generate common configuration as YAML anchor."""
        lines = [
            "x-common-config: &common-config",
            f"  volumes:",
            f"    - ${{HOME}}/{self.cache_hf}:/root/{self.cache_hf}",
            f"    - {self.data_dir}:/data",
        ]

        # Add driver library volume for manual mode
        if self.mount_mode == "manual" and self.driver_lib_dir:
            lines.append(f"    - {self.driver_lib_dir}:/usr/local/nvidia/lib64:ro")

        lines.extend(
            [
                f"  environment:",
                f"    - HF_ENDPOINT={self.hf_endpoint}",
                f"    - HF_HOME=/root/{self.cache_hf}",
                f"    - HF_HUB_CACHE=/root/{self.cache_hf_hub}",
                f"    - HUGGINGFACE_HUB_CACHE=/root/{self.cache_hf_hub}",
            ]
        )

        # Add HTTP proxy settings if specified
        if self.http_proxy:
            lines.extend(
                [
                    f"    - HTTP_PROXY={self.http_proxy}",
                    f"    - HTTPS_PROXY={self.http_proxy}",
                    f"    - http_proxy={self.http_proxy}",
                    f"    - https_proxy={self.http_proxy}",
                    # NO_PROXY for local services
                    f"    - NO_PROXY=localhost,127.0.0.1",
                    f"    - no_proxy=localhost,127.0.0.1",
                ]
            )

        # Add LD_LIBRARY_PATH for manual mode
        if self.mount_mode == "manual":
            lines.extend(
                [
                    f"    - LD_LIBRARY_PATH=/usr/local/nvidia/lib64",
                    f"    - CUDA_VISIBLE_DEVICES=0",
                    f"    - NVIDIA_VISIBLE_DEVICES=0",
                ]
            )

        lines.append(f"")
        return lines

    def _generate_service(self, gpu: GPUInfo) -> list[str]:
        """Generate service definition for a single GPU.

        Supports two modes:
        - nvidia-runtime: Uses Docker GPU reservation (requires nvidia-container-runtime)
        - manual: Manually mounts device nodes (bypasses nvidia-container-cli NVML detection)
        """
        gpu_config = self._get_gpu_config(gpu)
        service_port = self.port + gpu.index
        container_name = f"{self.project_name}--gpu{gpu.index}"

        lines = [
            f"  tei-gpu{gpu.index}:",
            f"    <<: *common-config",
            f"    image: {gpu.image}",
            f"    container_name: {container_name}",
        ]

        # Use host network when proxy is configured (to access host's proxy at 127.0.0.1)
        if self.http_proxy:
            lines.append(f"    network_mode: host")
            # In host mode, container uses host's network stack directly
            # No port mapping needed, service will be accessible at host:{service_port}
        else:
            # Bridge mode - need port mapping
            lines.extend(
                [
                    f"    ports:",
                    f'      - "{service_port}:80"',
                ]
            )

        if self.mount_mode == "manual":
            # Manual device mounting mode - bypasses nvidia-container-cli
            lines.append(f"    devices:")
            # Add GPU-specific device
            lines.append(f"      - /dev/nvidia{gpu.index}:/dev/nvidia{gpu.index}")
            # Add common NVIDIA devices
            for device in NvidiaDriverLibs.get_required_devices():
                if Path(device).exists():
                    lines.append(f"      - {device}:{device}")

        else:
            # nvidia-runtime mode - uses docker GPU reservation
            lines.extend(
                [
                    f"    deploy:",
                    f"      resources:",
                    f"        reservations:",
                    f"          devices:",
                    f"            - driver: nvidia",
                    f'              device_ids: ["{gpu.index}"]',
                    f"              capabilities: [gpu]",
                ]
            )

        # Generate command arguments
        lines.extend(
            [
                f"    command:",
                f"      - --huggingface-hub-cache",
                f"      - /root/{self.cache_hf_hub}",
                f"      - --model-id",
                f"      - {gpu_config.model_name}",
            ]
        )
        if self.hf_token:
            lines.extend(
                [
                    f"      - --hf-token",
                    f"      - {self.hf_token}",
                ]
            )
        lines.extend(
            [
                f"      - --dtype",
                f"      - float16",
                f"      - --max-batch-tokens",
                f'      - "32768"',
                f"      - --max-client-batch-size",
                f'      - "{MAX_CLIENT_BATCH_SIZE}"',
            ]
        )

        # In host network mode, specify the port for TEI to bind to
        if self.http_proxy:
            lines.extend(
                [
                    f"      - --port",
                    f'      - "{service_port}"',
                ]
            )

        # Generate healthcheck configuration
        lines.append(f"    healthcheck:")
        if self.http_proxy:
            # In host network mode, check the specific port TEI binds to
            lines.append(
                f'      test: ["CMD", "curl", "-f", "http://localhost:{service_port}/health"]'
            )
        else:
            # In bridge mode, always check port 80 (internal container port)
            lines.append(
                f'      test: ["CMD", "curl", "-f", "http://localhost:80/health"]'
            )

        lines.extend(
            [
                f"      interval: 30s",
                f"      timeout: 10s",
                f"      retries: 3",
                f"      start_period: 60s",
            ]
        )

        lines.append(f"")
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
        mount_mode: str = DEVICE_MOUNT_MODE,
        http_proxy: Optional[str] = None,
        gpu_configs: list[GpuModelConfig] | None = None,
    ):
        self.gpu_configs = gpu_configs or []
        configured_model_names = {
            config.model_name for config in self.gpu_configs if config.model_name
        }
        resolved_model_name = (model_name or MODEL_NAME).strip() or MODEL_NAME
        if len(configured_model_names) == 1:
            resolved_model_name = next(iter(configured_model_names))
        self.model_name = resolved_model_name
        self.port = port
        self.gpu_ids = infer_gpu_ids(gpu_ids, self.gpu_configs)
        self.hf_token = hf_token
        self.mount_mode = mount_mode
        self.http_proxy = http_proxy

        # project name must match: '^[a-z0-9][a-z0-9_-]*$'
        if project_name:
            self.project_name = project_name
        elif len(configured_model_names) > 1:
            self.project_name = "tei-multi"
        else:
            project_dash = self.model_name.replace("/", "--").lower()
            project_dash = re.sub(r"[^a-z0-9_-]", "_", project_dash)
            self.project_name = f"tei--{project_dash}"

        # Compose file location (default to configs directory)
        if compose_dir:
            self.compose_dir = Path(compose_dir)
        else:
            script_dir = Path(__file__).resolve().parent
            self.compose_dir = script_dir.parent / "configs"

        self.compose_file = self.compose_dir / f"{self.project_name}.yml"

        # Components
        self.gpus = GPUDetector.detect(self.gpu_ids)
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
            mount_mode=self.mount_mode,
            http_proxy=self.http_proxy,
            gpu_configs=self.gpu_configs,
        )
        content = compose_generator.generate()
        self.compose_file.write_text(content)
        logger.okay(f"[tfmx] Generated: {self.compose_file}")
        return self.compose_file

    def _configured_model_names(self) -> list[str]:
        if not self.gpu_configs:
            return [self.model_name]

        names: list[str] = []
        for config in self.gpu_configs:
            if config.model_name and config.model_name not in names:
                names.append(config.model_name)
        return names or [self.model_name]

    def get_backend_endpoints(self) -> list[str]:
        return [f"http://localhost:{self.port + gpu.index}" for gpu in self.gpus]

    def wait_for_healthy_backends(
        self,
        timeout_sec: float = 300.0,
        poll_interval_sec: float = 5.0,
        label: str = "[tei]",
    ) -> bool:
        return wait_for_healthy_http_endpoints(
            self.get_backend_endpoints(),
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            label=label,
        )

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

    def _start_all_together(self) -> tuple[bool, str]:
        """Try to start all services together.

        Returns:
            (success, message) tuple
        """
        try:
            # Use --remove-orphans to clean up containers from previous deployments
            # (e.g., when GPUs change or some GPUs are removed)
            result = self._run_compose_cmd(
                "up -d --remove-orphans", capture_output=True
            )
            if result.returncode != 0:
                error_msg = ""
                if result.stderr:
                    # Handle bytes or string
                    if isinstance(result.stderr, bytes):
                        error_msg = result.stderr.decode("utf-8", errors="replace")
                    else:
                        error_msg = result.stderr
                    error_msg = error_msg.strip()
                return False, error_msg or "Unknown error"

            # Return stdout for progress info display
            stdout_msg = ""
            if result.stdout:
                if isinstance(result.stdout, bytes):
                    stdout_msg = result.stdout.decode("utf-8", errors="replace")
                else:
                    stdout_msg = result.stdout
                stdout_msg = stdout_msg.strip()

            return True, stdout_msg or "All services started together"
        except Exception as e:
            return False, f"Exception: {e}"

    def up(self) -> None:
        """Start all TEI containers.

        Strategy:
        1. Pre-check all GPUs and filter unhealthy ones
        2. Start all services together
        3. Show diagnostics if startup fails
        """
        if not self.gpus:
            logger.warn("× No healthy GPUs detected")
            # Show diagnostic info
            gpu_summary = GPUDetector.get_unhealthy_gpu_summary()
            if gpu_summary:
                logger.warn(f"[tfmx] GPU issues detected:")
                for line in gpu_summary.split("\n"):
                    if line.strip():
                        logger.warn(f"  {line.strip()}")
            logger.hint("[tfmx] System may require reboot to recover dropped GPUs")
            return

        logger.mesg(f"[tfmx] Starting TEI for model: {self.model_name}")
        logger.mesg(
            f"[tfmx] GPUs: {[f'{g.index}(cap={g.compute_cap})' for g in self.gpus]}"
        )
        if self.gpu_configs:
            for config in self.gpu_configs:
                logger.mesg(f"[tfmx]   GPU {config.gpu_id}: {config.label}")

        # Patch config files (if cache exists and is accessible)
        for configured_model_name in self._configured_model_names():
            self.model_config_manager.patch_config_files(configured_model_name)

        # Ensure images are available
        images = set(g.image for g in self.gpus)
        for image in images:
            self.image_manager.ensure_image(image)

        # Ensure directories
        self._ensure_data_dir()

        # Generate compose file
        self.generate_compose_file()

        # Start all services together
        logger.mesg(f"[tfmx] Starting all services...")
        success, message = self._start_all_together()

        if success:
            # Show progress info
            if message and message != "All services started together":
                for line in message.split("\n"):
                    if line.strip():
                        logger.mesg(f" {line}")
            logger.okay(
                f"[tfmx] All GPU services started successfully: {[g.index for g in self.gpus]}"
            )
        else:
            # Parse error message and display with appropriate colors
            logger.warn(f"[tfmx] Startup failed")
            if message:
                error_found = False
                for line in message.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Check if line contains error indicators
                    if any(
                        keyword in line.lower()
                        for keyword in ["error", "failed", "unable"]
                    ):
                        logger.warn(f" {line}")
                        error_found = True
                    else:
                        # Normal progress messages
                        logger.mesg(f" {line}")

                # Show hint only if NVIDIA error detected
                if error_found and (
                    "nvml" in message.lower() or "nvidia" in message.lower()
                ):
                    logger.hint(
                        "[tfmx] NVIDIA runtime error - system may require reboot to recover dropped GPUs"
                    )

        # Show status
        self.ps()

    def down(self) -> None:
        """Stop and remove all TEI containers.

        This command works independently without requiring any parameters.
        It finds containers by project name prefix and removes them directly.
        """
        # Try compose file method first if it exists
        if self.compose_file.exists():
            logger.mesg(f"[tfmx] Using compose file: {self.compose_file}")
            self._run_compose_cmd("down --remove-orphans")
            return

        # Fallback: find and remove containers by name pattern
        logger.mesg(
            f"[tfmx] Compose file not found, searching for containers by pattern"
        )

        # Find all containers matching the project name pattern
        # Pattern: tei--<model>--gpu<N>
        pattern = f"{self.project_name}--gpu"
        result = subprocess.run(
            f'docker ps -a --filter "name={pattern}" --format "{{{{.Names}}}}"',
            shell=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.warn(f"× Failed to query containers")
            return

        container_names = [
            name.strip() for name in result.stdout.strip().split("\n") if name.strip()
        ]

        if not container_names:
            logger.mesg(f"[tfmx] No containers found matching pattern: {pattern}*")
            return

        logger.mesg(f"[tfmx] Found {len(container_names)} container(s) to remove")
        for name in container_names:
            logger.mesg(f"[tfmx]   - {name}")

        # Remove all matching containers
        for name in container_names:
            subprocess.run(
                f"docker rm -f {name}",
                shell=True,
                capture_output=True,
            )

        logger.okay(f"[tfmx] Removed {len(container_names)} container(s)")

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

    def setup(self) -> None:
        """Setup model cache with required config files.

        This creates sentence_*_config.json files in the model cache directory
        to prevent TEI from trying to download them (which is slow due to network restrictions).

        Uses a Docker container to write files with correct permissions (root).
        Run this once before first 'up' to avoid slow downloads.
        """
        logger.mesg(f"[tfmx] Setting up model cache for: {self.model_name}")

        # Build the model cache path pattern
        model_name_dash = self.model_name.replace("/", "--")
        cache_path = f"/root/{CACHE_HF_HUB}/models--{model_name_dash}/snapshots"

        # Build list of config files for shell
        config_files_str = " ".join(SENTENCE_CONFIG_FILES)

        # Build the shell script to create config files
        # Note: sentence_bert_config.json requires specific fields
        create_files_script = f"""
SNAPSHOT_DIR=$(find {cache_path} -maxdepth 1 -type d 2>/dev/null | tail -n 1)

if [ -z "$SNAPSHOT_DIR" ] || [ "$SNAPSHOT_DIR" = "{cache_path}" ]; then
    echo "[setup] Error: Model snapshot not found at {cache_path}"
    echo "[setup] Please download the model first with: huggingface-cli download {self.model_name}"
    exit 1
fi

echo "[setup] Found snapshot: $SNAPSHOT_DIR"

# Create sentence_bert_config.json with required fields
if [ ! -f "$SNAPSHOT_DIR/sentence_bert_config.json" ]; then
    cat > "$SNAPSHOT_DIR/sentence_bert_config.json" << 'EOF'
{{
  "max_seq_length": 256,
  "do_lower_case": false
}}
EOF
    echo "[setup] Created: sentence_bert_config.json"
else
    echo "[setup] Skip existed: sentence_bert_config.json"
fi

# Create other sentence_*_config.json files (empty JSON is fine for these)
for config_file in sentence_roberta_config.json sentence_distilbert_config.json sentence_camembert_config.json sentence_albert_config.json sentence_xlm-roberta_config.json sentence_xlnet_config.json; do
    target="$SNAPSHOT_DIR/$config_file"
    if [ -f "$target" ]; then
        echo "[setup] Skip existed: $config_file"
    else
        echo "{{}}" > "$target"
        echo "[setup] Created: $config_file"
    fi
done

echo "[setup] Setup completed successfully!"
"""

        # Get any healthy GPU's image for running the setup container
        if self.gpus:
            image = self.gpus[0].image
        else:
            # Fallback to default image
            image = f"{TEI_IMAGE_BASE}:{TEI_TAG}"

        # Ensure image is available
        self.image_manager.ensure_image(image)

        # Run setup in a Docker container
        # We need to mount the huggingface cache directory
        docker_cmd = f"""docker run --rm \\
            -v "${{HOME}}/{CACHE_HF}:/root/{CACHE_HF}" \\
            --entrypoint /bin/sh \\
            {image} \\
            -c '{create_files_script}'
"""

        logger.mesg(f"[tfmx] Running setup container...")

        result = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True)

        # Show output
        if result.stdout:
            for line in result.stdout.strip().split("\\n"):
                if line.strip():
                    if "Error" in line:
                        logger.warn(line)
                    elif "Skip" in line:
                        logger.mesg(line)
                    elif "Created" in line:
                        logger.okay(line)
                    else:
                        logger.mesg(line)

        if result.returncode != 0:
            if result.stderr:
                for line in result.stderr.strip().split("\\n"):
                    if line.strip():
                        logger.warn(f"  {line}")
            logger.warn("[tfmx] Setup failed")
        else:
            logger.okay("[tfmx] Model cache setup completed")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=SERVER_PORT,
        help=f"Starting port (default: {SERVER_PORT})",
    )
    parser.add_argument(
        "-j",
        "--project-name",
        type=str,
        default=None,
        help="Project name (default: tei--MODEL_NAME)",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (default: all healthy GPUs)",
    )
    parser.add_argument(
        "--gpu-configs",
        type=str,
        default=None,
        help='Per-GPU config: "GPU[:MODEL],..."',
    )


def _add_deployment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for private models",
    )
    parser.add_argument(
        "--mount-mode",
        type=str,
        choices=["nvidia-runtime", "manual"],
        default=DEVICE_MOUNT_MODE,
        help=f"Device mount mode (default: {DEVICE_MOUNT_MODE})",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP/HTTPS proxy for model downloads (e.g., $TEI_PROXY_URL)",
    )


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Manage TEI docker compose deployments"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = CLI_EPILOG

    subparsers = parser.add_subparsers(dest="compose_action", required=False)

    parser_up = subparsers.add_parser("up", help="Start TEI containers")
    _add_common_arguments(parser_up)
    _add_deployment_arguments(parser_up)

    for action in ("down", "stop", "start", "restart", "ps", "health"):
        action_parser = subparsers.add_parser(
            action, help=f"{action.title()} TEI containers"
        )
        _add_common_arguments(action_parser)

    parser_logs = subparsers.add_parser("logs", help="View TEI container logs")
    _add_common_arguments(parser_logs)
    parser_logs.add_argument("-f", "--follow", action="store_true")
    parser_logs.add_argument("--tail", type=int, default=100)

    parser_generate = subparsers.add_parser(
        "generate", help="Generate docker-compose.yml only"
    )
    _add_common_arguments(parser_generate)
    _add_deployment_arguments(parser_generate)

    parser_setup = subparsers.add_parser("setup", help="Setup model cache config files")
    _add_common_arguments(parser_setup)
    _add_deployment_arguments(parser_setup)


def _composer_from_args(args: argparse.Namespace) -> TEIComposer:
    model_name = getattr(args, "model_name", MODEL_NAME)
    if model_name:
        model_name = model_name.strip() or MODEL_NAME
    else:
        model_name = MODEL_NAME

    composer_kwargs = {
        "model_name": model_name,
        "port": getattr(args, "port", SERVER_PORT),
        "project_name": getattr(args, "project_name", None),
        "gpu_ids": getattr(args, "gpus", None),
        "hf_token": getattr(args, "hf_token", None),
        "mount_mode": getattr(args, "mount_mode", DEVICE_MOUNT_MODE),
        "http_proxy": getattr(args, "proxy", None),
    }
    gpu_configs = getattr(args, "gpu_configs", None)
    if gpu_configs:
        composer_kwargs["gpu_configs"] = parse_gpu_configs(gpu_configs)
    return TEIComposer(**composer_kwargs)


def run_from_args(args: argparse.Namespace) -> None:
    action = getattr(args, "compose_action", None)
    if not action:
        raise ValueError("Compose action is required")

    composer = _composer_from_args(args)

    if action == "up":
        composer.up()
    elif action == "down":
        composer.down()
    elif action == "stop":
        composer.stop()
    elif action == "start":
        composer.start()
    elif action == "restart":
        composer.restart()
    elif action == "ps":
        composer.ps()
    elif action == "logs":
        composer.logs(
            follow=getattr(args, "follow", False), tail=getattr(args, "tail", 100)
        )
    elif action == "generate":
        composer.generate_compose_file()
    elif action == "health":
        composer.health()
    elif action == "setup":
        composer.setup()
    else:
        raise ValueError(f"Unknown compose action: {action}")


class TEIComposeArgParser:
    """Compatibility wrapper around the reusable compose parser."""

    def __init__(self, argv: list[str] | None = None):
        self.parser = argparse.ArgumentParser()
        configure_parser(self.parser)
        self.args = self.parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arg_parser = TEIComposeArgParser(argv)
    args = arg_parser.args

    if not getattr(args, "compose_action", None):
        arg_parser.parser.print_help()
        return

    run_from_args(args)


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_compose.py#clis
