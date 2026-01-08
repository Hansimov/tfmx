"""TEI (Text Embeddings Inference) Docker Compose Manager

This module provides low-level Docker Compose operations for TEI containers.
For user-friendly server management, use 'tei_server' command instead.

This tool is designed for:
- Advanced Docker Compose operations
- Direct control over compose lifecycle
- Custom deployment workflows
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from tclogger import logger


# ============================================================================
# Constants
# ============================================================================

SERVER_PORT = 28880
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
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


# ============================================================================
# GPU Information
# ============================================================================


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


# ============================================================================
# Configuration Management
# ============================================================================


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


# ============================================================================
# Docker Image Management
# ============================================================================


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


# ============================================================================
# Docker Compose File Generator
# ============================================================================


class ComposeFileGenerator:
    """Generates docker-compose.yml content for TEI deployment."""

    def __init__(
        self,
        cache_hf: str = CACHE_HF,
        cache_hf_hub: str = CACHE_HF_HUB,
        hf_endpoint: str = HF_ENDPOINT,
    ):
        self.cache_hf = cache_hf
        self.cache_hf_hub = cache_hf_hub
        self.hf_endpoint = hf_endpoint

    def generate(
        self,
        gpus: list[GPUInfo],
        model_name: str,
        port: int,
        project_name: str,
        data_dir: Path,
        hf_token: Optional[str] = None,
    ) -> str:
        """Generate docker-compose.yml content."""
        lines = self._generate_header(model_name, project_name, gpus)
        lines.append("services:")

        for gpu in gpus:
            service_lines = self._generate_service(
                gpu=gpu,
                model_name=model_name,
                port=port,
                project_name=project_name,
                data_dir=data_dir,
                hf_token=hf_token,
            )
            lines.extend(service_lines)

        return "\n".join(lines)

    def _generate_header(
        self, model_name: str, project_name: str, gpus: list[GPUInfo]
    ) -> list[str]:
        """Generate compose file header."""
        return [
            f"# TEI Multi-GPU Deployment",
            f"# Model: {model_name}",
            f"# GPUs: {[g.index for g in gpus]}",
            f"",
            f"name: {project_name}",
            f"",
        ]

    def _generate_service(
        self,
        gpu: GPUInfo,
        model_name: str,
        port: int,
        project_name: str,
        data_dir: Path,
        hf_token: Optional[str] = None,
    ) -> list[str]:
        """Generate service definition for a single GPU."""
        service_port = port + gpu.index
        container_name = f"{project_name}--gpu{gpu.index}"

        lines = [
            f"  tei-gpu{gpu.index}:",
            f"    image: {gpu.image}",
            f"    container_name: {container_name}",
            f"    ports:",
            f'      - "{service_port}:80"',
            f"    volumes:",
            f"      - ${{HOME}}/{self.cache_hf}:/root/{self.cache_hf}",
            f"      - {data_dir}:/data",
            f"    environment:",
            f"      - HF_ENDPOINT={self.hf_endpoint}",
            f"      - HF_HOME=/root/{self.cache_hf}",
            f"      - HF_HUB_CACHE=/root/{self.cache_hf_hub}",
            f"      - HUGGINGFACE_HUB_CACHE=/root/{self.cache_hf_hub}",
            f"    command:",
            f"      - --huggingface-hub-cache",
            f"      - /root/{self.cache_hf_hub}",
            f"      - --model-id",
            f"      - {model_name}",
        ]

        if hf_token:
            lines.extend([f"      - --hf-token", f"      - {hf_token}"])

        lines.extend(
            [
                f"      - --dtype",
                f"      - float16",
                f"      - --max-batch-tokens",
                f'      - "32768"',
                f"      - --max-client-batch-size",
                f'      - "100"',
                f"    deploy:",
                f"      resources:",
                f"        reservations:",
                f"          devices:",
                f"            - driver: nvidia",
                f'              device_ids: ["{gpu.index}"]',
                f"              capabilities: [gpu]",
                f"    restart: unless-stopped",
                f"    healthcheck:",
                f'      test: ["CMD", "curl", "-f", "http://localhost:80/health"]',
                f"      interval: 30s",
                f"      timeout: 10s",
                f"      retries: 3",
                f"      start_period: 60s",
                f"",
            ]
        )

        return lines


# ============================================================================
# TEI Compose Manager
# ============================================================================


class TEIComposeManager:
    """Manager for TEI Docker Compose deployments."""

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

        # Project name from model (must be lowercase, alphanumeric with - and _)
        model_dash = model_name.replace("/", "--").lower()
        self.project_name = project_name or f"tei--{model_dash}"

        # Compose file location
        if compose_dir:
            self.compose_dir = Path(compose_dir)
        else:
            self.compose_dir = Path.home() / ".tfmx" / "compose"

        self.compose_file = self.compose_dir / f"{self.project_name}.yml"

        # Components
        self.gpus = GPUDetector.detect(gpu_ids)
        self.model_config_manager = ModelConfigManager()
        self.image_manager = DockerImageManager()
        self.compose_generator = ComposeFileGenerator()

    def _ensure_compose_dir(self) -> None:
        """Ensure compose directory exists."""
        self.compose_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_data_dir(self) -> Path:
        """Ensure docker_data directory exists."""
        data_dir = Path.home() / ".tfmx" / "docker_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def generate_compose_file(self) -> Path:
        """Generate the docker-compose.yml file."""
        self._ensure_compose_dir()
        data_dir = self._ensure_data_dir()

        content = self.compose_generator.generate(
            gpus=self.gpus,
            model_name=self.model_name,
            port=self.port,
            project_name=self.project_name,
            data_dir=data_dir,
            hf_token=self.hf_token,
        )

        self.compose_file.write_text(content)
        logger.okay(f"[tfmx] Generated: {self.compose_file}")
        return self.compose_file

    def _run_compose_cmd(self, cmd: str) -> subprocess.CompletedProcess:
        """Run a docker compose command."""
        full_cmd = f"docker compose -f {self.compose_file} {cmd}"
        logger.mesg(f"[tfmx] Running: {full_cmd}")
        return subprocess.run(full_cmd, shell=True)

    def up(self) -> None:
        """Start all TEI containers."""
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

        # Generate and run
        self.generate_compose_file()
        self._run_compose_cmd("up -d")

        # Show status
        self.ps()

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
        print("=" * 70)
        print(f"{'GPU':<6} {'CONTAINER':<40} {'PORT':<8} {'STATUS':<10}")
        print("-" * 70)

        for gpu in self.gpus:
            container_name = f"{self.project_name}--gpu{gpu.index}"
            container_port = self.port + gpu.index

            # Check container status
            result = subprocess.run(
                f'docker ps -a --filter "name=^/{container_name}$" --format "{{{{.Status}}}}"',
                shell=True,
                capture_output=True,
                text=True,
            )
            status = result.stdout.strip() or "not found"
            if status.startswith("Up"):
                status = "running"
            elif status.startswith("Exited"):
                status = "stopped"

            print(
                f"{gpu.index:<6} {container_name:<40} {container_port:<8} {status:<10}"
            )

        print("=" * 70)


# ============================================================================
# Argument Parser
# ============================================================================


class TEIComposeArgParser:
    """Argument parser for TEI Compose CLI."""

    # ANCHOR[id=clis]
    EPILOG = """
Examples:
  # Set model as environment variable for convenience
  export MODEL="Alibaba-NLP/gte-multilingual-base"
  
  tei_compose -m "$MODEL" up           # Start on all GPUs
  tei_compose -m "$MODEL" -g "0,1" up  # Start on specific GPUs
  tei_compose -m "$MODEL" ps           # Check status
  tei_compose -m "$MODEL" logs -f      # View logs
  tei_compose -m "$MODEL" stop         # Stop containers
  tei_compose -m "$MODEL" down         # Remove containers
    """

    def __init__(self):
        import argparse

        self.parser = argparse.ArgumentParser(
            description="TEI Docker Compose Manager (Low-level operations)\n\nFor user-friendly operations, use 'tei_server' command instead.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        """Setup all command-line arguments."""
        # Model configuration
        self.parser.add_argument(
            "-m",
            "--model",
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
            "-n",
            "--name",
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
            "-u",
            "--hf-token",
            type=str,
            default=None,
            help="HuggingFace token for private models",
        )

        # Actions
        self.parser.add_argument(
            "action",
            choices=[
                "up",
                "down",
                "stop",
                "start",
                "restart",
                "ps",
                "logs",
                "generate",
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


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    args = TEIComposeArgParser().args

    manager = TEIComposeManager(
        model_name=args.model,
        port=args.port,
        project_name=args.name,
        gpu_ids=args.gpus,
        hf_token=args.hf_token,
    )

    if args.action == "up":
        manager.up()
    elif args.action == "down":
        manager.down()
    elif args.action == "stop":
        manager.stop()
    elif args.action == "start":
        manager.start()
    elif args.action == "restart":
        manager.restart()
    elif args.action == "ps":
        manager.ps()
    elif args.action == "logs":
        manager.logs(follow=args.follow, tail=args.tail)
    elif args.action == "generate":
        manager.generate_compose_file()


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_compose.py#clis
