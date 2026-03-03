"""QVL (Qwen3-VL) Docker Compose Manager

Manages Docker Compose deployments for Qwen3-VL vision-language models
running on vLLM inference engine with AWQ quantization support.
"""

# ANCHOR[id=clis]
CLI_EPILOG = """
Examples:
  # Set model as environment variable for convenience
  export MODEL="Qwen/Qwen3-VL-8B-Instruct"

  # Basic operations
  qvl_compose up                    # Start on all GPUs
  qvl_compose ps                    # Check container status
  qvl_compose logs                  # View recent logs
  qvl_compose stop                  # Stop containers (keep them)
  qvl_compose start                 # Start stopped containers
  qvl_compose restart               # Restart containers
  qvl_compose down                  # Stop and remove containers
  qvl_compose generate              # Generate compose file only
  qvl_compose health                # Check GPU health status

  # With specific model
  qvl_compose up -m "$MODEL"
  qvl_compose up -m "Qwen/Qwen3-VL-4B-Instruct"

  # With AWQ quantization (recommended for RTX 30/40)
  qvl_compose up -m "Qwen/Qwen3-VL-8B-Instruct" -q awq

  # With specific GPUs
  qvl_compose up -g "0,1"           # Start on GPU 0 and 1
  qvl_compose up -g "2"             # Start on GPU 2 only

  # Custom port and project name
  qvl_compose up -p 29890           # Use port 29890 as base
  qvl_compose up -j my-qvl          # Custom project name

  # With HTTP proxy for downloading models
  qvl_compose up --proxy http://127.0.0.1:11111

  # Advanced: Manual device mount mode
  qvl_compose up --mount-mode manual

  # Advanced log viewing
  qvl_compose logs -f               # Follow logs in real-time
  qvl_compose logs --tail 200       # Show last 200 lines

  # Per-GPU model/quant config (AWQ 4bit)
  qvl_compose up --gpu-configs "0:2b-instruct:4bit,1:8b-instruct:4bit"

Supported Models:
  - Qwen/Qwen3-VL-2B-Instruct      (2B parameters, instruction-tuned)
  - Qwen/Qwen3-VL-2B-Thinking      (2B parameters, thinking mode)
  - Qwen/Qwen3-VL-4B-Instruct      (4B parameters, instruction-tuned)
  - Qwen/Qwen3-VL-4B-Thinking      (4B parameters, thinking mode)
  - Qwen/Qwen3-VL-8B-Instruct      (8B parameters, instruction-tuned)
  - Qwen/Qwen3-VL-8B-Thinking      (8B parameters, thinking mode)

Quantization Options (for RTX 30/40 series):
  - awq:          AWQ quantization from cyankiwi (4bit)
  - bitsandbytes: BitsAndBytes 4-bit quantization
  - none:         No quantization (requires more VRAM)

Device Mount Modes:
  nvidia-runtime: (default) Uses Docker GPU reservation
  manual:         Manually mounts /dev/nvidia* device nodes
"""

import argparse
import re
import subprocess

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tclogger import logger


SERVER_PORT = 29880
MACHINE_PORT = 29800
MODEL_NAME = "Qwen/Qwen3-VL-4B-Thinking"
HF_ENDPOINT = "https://hf-mirror.com"
CACHE_HF = ".cache/huggingface"
CACHE_HF_HUB = f"{CACHE_HF}/hub"

# vLLM image
VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_IMAGE_MIRROR = "m.daocloud.io"

# vLLM internal port (default)
VLLM_INTERNAL_PORT = 8000

# Max concurrent requests per vLLM instance
MAX_CONCURRENT_REQUESTS = 8

# vLLM model parameters
MAX_MODEL_LEN = 8096
MAX_NUM_SEQS = 5
LIMIT_MM_PER_PROMPT = '{"image": 5}'
GPU_MEMORY_UTILIZATION = 0.40

# Per-model-size GPU memory utilization limits (fraction of total VRAM)
# Designed for RTX 3080 20GB with TEI colocated (~1.4GB each)
# 2B: ≤6GB → 0.30, 4B: ≤10GB → 0.50, 8B: ≤14GB → 0.70
GPU_MEMORY_UTILIZATION_BY_SIZE = {
    "2B": 0.30,
    "4B": 0.50,
    "8B": 0.70,
}

# Device mount mode
DEVICE_MOUNT_MODE = "manual"

# Supported models
SUPPORTED_MODELS = {
    "Qwen/Qwen3-VL-2B-Instruct": {
        "size": "2B",
        "type": "instruct",
        "max_model_len": 8096,
    },
    "Qwen/Qwen3-VL-2B-Thinking": {
        "size": "2B",
        "type": "thinking",
        "max_model_len": 8096,
    },
    "Qwen/Qwen3-VL-4B-Instruct": {
        "size": "4B",
        "type": "instruct",
        "max_model_len": 8096,
    },
    "Qwen/Qwen3-VL-4B-Thinking": {
        "size": "4B",
        "type": "thinking",
        "max_model_len": 8096,
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "size": "8B",
        "type": "instruct",
        "max_model_len": 8096,
    },
    "Qwen/Qwen3-VL-8B-Thinking": {
        "size": "8B",
        "type": "thinking",
        "max_model_len": 8096,
    },
}

# AWQ quantized variants from cyankiwi (4bit only)
# Format: cyankiwi/Qwen3-VL-{size}-{type}-AWQ-4bit
AWQ_MODELS = {
    "cyankiwi/Qwen3-VL-2B-Instruct-AWQ-4bit": "Qwen/Qwen3-VL-2B-Instruct",
    "cyankiwi/Qwen3-VL-2B-Thinking-AWQ-4bit": "Qwen/Qwen3-VL-2B-Thinking",
    "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit": "Qwen/Qwen3-VL-4B-Instruct",
    "cyankiwi/Qwen3-VL-4B-Thinking-AWQ-4bit": "Qwen/Qwen3-VL-4B-Thinking",
    "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit": "Qwen/Qwen3-VL-8B-Instruct",
    "cyankiwi/Qwen3-VL-8B-Thinking-AWQ-4bit": "Qwen/Qwen3-VL-8B-Thinking",
}

# Model shortcuts: size-type → full HF model name
# Keys are lowercase for case-insensitive matching
MODEL_SHORTCUTS = {
    "2b-instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "2b-thinking": "Qwen/Qwen3-VL-2B-Thinking",
    "4b-instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "4b-thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "8b-instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "8b-thinking": "Qwen/Qwen3-VL-8B-Thinking",
}

# Reverse map: full name → shortcut (lowercase)
MODEL_SHORTCUT_REV = {v: k for k, v in MODEL_SHORTCUTS.items()}
# Also map lowercased full names for case-insensitive lookup
MODEL_SHORTCUT_REV_LOWER = {v.lower(): k for k, v in MODEL_SHORTCUTS.items()}

# Original (display) shortcut names: lowercase key → display name
_DISPLAY_SHORTCUTS = {
    "2b-instruct": "2B-Instruct",
    "2b-thinking": "2B-Thinking",
    "4b-instruct": "4B-Instruct",
    "4b-thinking": "4B-Thinking",
    "8b-instruct": "8B-Instruct",
    "8b-thinking": "8B-Thinking",
}

# Base model → AWQ repo mapping: (shortcut, quant_level) → cyankiwi repo
# Uses _DISPLAY_SHORTCUTS to generate display-cased repo names (4bit only)
AWQ_REPO_MAP = {
    (shortcut, "4bit"): f"cyankiwi/Qwen3-VL-{_DISPLAY_SHORTCUTS[shortcut]}-AWQ-4bit"
    for shortcut in MODEL_SHORTCUTS
}

# Default AWQ configuration (all lowercase for consistency)
DEFAULT_QUANT_METHOD = "awq"
DEFAULT_QUANT_LEVEL = "4bit"

# Known AWQ quant levels (lowercase)
AWQ_QUANT_LEVELS = {"4bit"}


def normalize_model_key(key: str) -> str:
    """Normalize a model shortcut or name to lowercase for matching.

    Examples:
        normalize_model_key("8B-Instruct") -> "8b-instruct"
        normalize_model_key("4bit") -> "4bit"
        normalize_model_key("Qwen/Qwen3-VL-8B-Instruct") -> "qwen/qwen3-vl-8b-instruct"
    """
    return key.strip().lower() if key else ""


def resolve_model_name(key: str) -> str:
    """Resolve a model key (shortcut or full name) to the canonical full HF name.

    Case-insensitive. Returns original key if no match found.

    Examples:
        resolve_model_name("8b-instruct") -> "Qwen/Qwen3-VL-8B-Instruct"
        resolve_model_name("8B-Instruct") -> "Qwen/Qwen3-VL-8B-Instruct"
        resolve_model_name("Qwen/Qwen3-VL-8B-Instruct") -> "Qwen/Qwen3-VL-8B-Instruct"
    """
    normalized = normalize_model_key(key)
    # Check shortcuts first
    if normalized in MODEL_SHORTCUTS:
        return MODEL_SHORTCUTS[normalized]
    # Check if it's already a full name (case-insensitive)
    for full_name in SUPPORTED_MODELS:
        if full_name.lower() == normalized:
            return full_name
    return key


def resolve_quant_level(level: str) -> str:
    """Normalize quant level to lowercase.

    Examples:
        resolve_quant_level("4bit") -> "4bit"
        resolve_quant_level("4BIT") -> "4bit"
    """
    return normalize_model_key(level)


def get_model_shortcut(model_name: str) -> str:
    """Get the lowercase shortcut for a model name.

    Supports both original HF names and AWQ cyankiwi repo names.

    Examples:
        get_model_shortcut("Qwen/Qwen3-VL-8B-Instruct") -> "8b-instruct"
        get_model_shortcut("cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit") -> "8b-instruct"
    """
    result = MODEL_SHORTCUT_REV.get(model_name)
    if result:
        return result
    # Try case-insensitive lookup
    result = MODEL_SHORTCUT_REV_LOWER.get(model_name.lower())
    if result:
        return result
    # Try AWQ model name → base model → shortcut
    base_model = AWQ_MODELS.get(model_name)
    if base_model:
        result = MODEL_SHORTCUT_REV.get(base_model)
        if result:
            return result
    # Case-insensitive AWQ lookup
    name_lower = model_name.lower()
    for awq_name, base in AWQ_MODELS.items():
        if awq_name.lower() == name_lower:
            result = MODEL_SHORTCUT_REV.get(base)
            if result:
                return result
    # Fallback: extract from full name
    parts = model_name.split("/")[-1] if model_name else ""
    return parts.lower()


def get_display_shortcut(shortcut: str) -> str:
    """Get display-friendly shortcut (original casing).

    Examples:
        get_display_shortcut("8b-instruct") -> "8B-Instruct"
    """
    return _DISPLAY_SHORTCUTS.get(normalize_model_key(shortcut), shortcut)


# Quantization recommendations by GPU VRAM and model size
# RTX 3060/3070: ~8-12GB, RTX 3080/3090: ~10-24GB
# RTX 4060/4070: ~8-12GB, RTX 4080/4090: ~16-24GB
QUANT_RECOMMENDATIONS = {
    "2B": {"min_vram_gb": 4, "recommended_quant": "none"},
    "4B": {"min_vram_gb": 6, "recommended_quant": "awq"},
    "8B": {"min_vram_gb": 10, "recommended_quant": "awq"},
}

# GPU compute capability to image tag mapping (vLLM uses single universal image)
# But we track compute caps for compatibility checks
GPU_COMPUTE_CAPS = {
    "8.6": "RTX 30xx",  # Ampere 86 (RTX 3080, 3090)
    "8.9": "RTX 40xx",  # Ada Lovelace (RTX 4090)
    "8.0": "A100/A30",  # Ampere 80
    "9.0": "H100",  # Hopper
}


@dataclass
class GpuModelConfig:
    """Per-GPU model and quantization configuration.

    All string fields are stored in normalized form:
    - model_name: canonical HF name (e.g., "Qwen/Qwen3-VL-8B-Instruct")
    - quant_method: lowercase (e.g., "awq")
    - quant_level: lowercase (e.g., "4bit")
    """

    gpu_id: int
    model_name: str = MODEL_NAME
    quant_method: str = DEFAULT_QUANT_METHOD
    quant_level: str = DEFAULT_QUANT_LEVEL

    def __post_init__(self):
        """Normalize model name and quant fields."""
        self.model_name = resolve_model_name(self.model_name)
        self.quant_method = self.quant_method.lower().strip()
        self.quant_level = self.quant_level.lower().strip()

    @property
    def model_shortcut(self) -> str:
        """Get lowercase model shortcut from full name."""
        return get_model_shortcut(self.model_name)

    @property
    def display_shortcut(self) -> str:
        """Get display-friendly shortcut (original casing)."""
        return get_display_shortcut(self.model_shortcut)

    @property
    def model_size(self) -> str:
        """Get model size (e.g., '2B', '4B', '8B') from model name."""
        info = SUPPORTED_MODELS.get(self.model_name)
        if info:
            return info["size"]
        # Try AWQ lookup
        base = AWQ_MODELS.get(self.model_name)
        if base:
            info = SUPPORTED_MODELS.get(base)
            if info:
                return info["size"]
        # Fallback: extract from shortcut (e.g., '8b-instruct' → '8B')
        sc = self.model_shortcut
        for prefix in ("2b", "4b", "8b"):
            if sc.startswith(prefix):
                return prefix.upper()
        return "8B"  # conservative default

    @property
    def gpu_memory_utilization(self) -> float:
        """Get GPU memory utilization for this model size."""
        return GPU_MEMORY_UTILIZATION_BY_SIZE.get(
            self.model_size, GPU_MEMORY_UTILIZATION
        )

    @property
    def awq_repo(self) -> str | None:
        """Get AWQ repo name for this model and quant level."""
        if self.quant_method != "awq":
            return None
        shortcut = self.model_shortcut
        return AWQ_REPO_MAP.get((shortcut, self.quant_level))

    @property
    def vllm_model_arg(self) -> str:
        """Get the model argument for vLLM serve command.

        For AWQ models, returns the cyankiwi AWQ repo name
        (e.g. ``cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit``).
        vLLM auto-detects ``compressed-tensors`` from the model config.
        """
        if self.quant_method == "awq":
            repo = self.awq_repo
            if repo:
                return repo
        return self.model_name

    @property
    def label(self) -> str:
        """Human-readable label (lowercase)."""
        if self.quant_level:
            return f"{self.model_shortcut}:{self.quant_level}"
        return self.model_shortcut

    @property
    def display_label(self) -> str:
        """Display-friendly label with original casing."""
        if self.quant_level:
            return f"{self.display_shortcut}:{self.quant_level.upper()}"
        return self.display_shortcut

    def to_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "model_name": self.model_name,
            "model_shortcut": self.model_shortcut,
            "quant_method": self.quant_method,
            "quant_level": self.quant_level,
            "awq_repo": self.awq_repo,
        }


def parse_gpu_configs(config_str: str) -> list[GpuModelConfig]:
    """Parse per-GPU model/quant configs from CLI string.

    Format: "GPU_ID:MODEL_SHORTCUT:QUANT_LEVEL,..."
    Example: "0:2b-instruct:4bit,1:4b-instruct:4bit,2:8b-instruct:4bit"

    Case-insensitive: '8B-Instruct' and '8b-instruct' are equivalent.
    If QUANT_LEVEL is omitted, defaults to 4bit.
    MODEL_SHORTCUT can be a shortcut (e.g., '8b-instruct') or full name.
    """
    configs = []
    for part in config_str.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) < 2:
            raise ValueError(f"Invalid config: '{part}'. Format: GPU_ID:MODEL[:QUANT]")

        gpu_id = int(fields[0].strip())
        model_key = normalize_model_key(fields[1])
        quant_level = (
            normalize_model_key(fields[2]) if len(fields) > 2 else DEFAULT_QUANT_LEVEL
        )

        # Resolve model name (shortcut -> full HF name)
        model_name = resolve_model_name(model_key)

        # Determine quant method from quant level
        quant_method = "awq" if quant_level in AWQ_QUANT_LEVELS else "none"

        configs.append(
            GpuModelConfig(
                gpu_id=gpu_id,
                model_name=model_name,
                quant_method=quant_method,
                quant_level=quant_level,
            )
        )

    return configs


class NvidiaDriverLibs:
    """Detect and manage NVIDIA driver library paths."""

    @staticmethod
    def detect_driver_lib_dir() -> Optional[str]:
        """Detect NVIDIA driver library directory."""
        candidates = [
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/local/nvidia/lib64",
        ]

        for path in candidates:
            if Path(path).exists():
                if list(Path(path).glob("libcuda.so*")):
                    return path

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
        """Get list of required NVIDIA device nodes for manual mounting."""
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
        self.arch_name = GPU_COMPUTE_CAPS.get(compute_cap, "Unknown")
        self.image = VLLM_IMAGE  # vLLM uses universal image

    def __repr__(self):
        return f"GPU({self.index}, cap={self.compute_cap}, arch={self.arch_name})"


class GPUDetector:
    """GPU detection and management."""

    @staticmethod
    def detect(
        gpu_ids: Optional[str] = None, check_health: bool = True
    ) -> list[GPUInfo]:
        """Detect available GPUs and their compute capabilities."""
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
            )

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

            if result.stderr:
                stderr_lower = result.stderr.lower()
                if "error" in stderr_lower or "unable" in stderr_lower:
                    logger.warn(f"[qvl] nvidia-smi warnings detected:")
                    for line in result.stderr.strip().split("\n"):
                        if line.strip():
                            logger.warn(f"  {line.strip()}")

            if gpu_ids:
                specified = [int(x.strip()) for x in gpu_ids.split(",")]
                gpus = [g for g in gpus if g.index in specified]

            if check_health:
                healthy_gpus = []
                for gpu in gpus:
                    is_healthy, msg = GPUDetector.check_gpu_health(gpu.index)
                    if is_healthy:
                        healthy_gpus.append(gpu)
                    else:
                        unhealthy_gpus.append((gpu.index, msg))
                        logger.warn(
                            f"[qvl] GPU {gpu.index} excluded (unhealthy): {msg}"
                        )
                gpus = healthy_gpus

            return gpus
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warn(f"× Failed to detect GPUs: {e}")
            return []

    @staticmethod
    def check_gpu_health(gpu_index: int) -> tuple[bool, str]:
        """Check if a specific GPU is healthy and accessible."""
        try:
            result = subprocess.run(
                f"nvidia-smi -i {gpu_index} --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )

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

            stdout_lower = result.stdout.lower()
            if "error" in stdout_lower or "failed" in stdout_lower:
                return False, f"GPU error: {result.stdout.strip()}"

            if not result.stdout.strip():
                return False, "No GPU info returned"

            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "nvidia-smi timeout (GPU may be frozen)"
        except Exception as e:
            return False, f"Health check failed: {e}"

    @staticmethod
    def get_unhealthy_gpu_summary() -> str:
        """Get a summary of any unhealthy GPUs."""
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
    """Manages model configuration for Qwen3-VL models."""

    def __init__(self, cache_hf_hub: str = CACHE_HF_HUB):
        self.cache_hf_hub = cache_hf_hub

    def get_model_snapshot_dir(self, model_name: str) -> Optional[Path]:
        """Find the model snapshot directory in HuggingFace cache."""
        model_name_dash = model_name.replace("/", "--")
        cache_path = Path.home() / self.cache_hf_hub

        if not cache_path.exists():
            return None

        model_dir = cache_path / f"models--{model_name_dash}"
        if not model_dir.exists():
            return None

        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                return snapshot

        return None

    def get_model_config(self, model_name: str) -> dict:
        """Get model-specific configuration."""
        if model_name in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_name]

        # Check AWQ models
        base_model = AWQ_MODELS.get(model_name)
        if base_model and base_model in SUPPORTED_MODELS:
            config = SUPPORTED_MODELS[base_model].copy()
            config["awq"] = True
            config["base_model"] = base_model
            return config

        # Default config
        return {
            "size": "unknown",
            "type": "instruct",
            "max_model_len": MAX_MODEL_LEN,
        }

    def get_quantization_recommendation(
        self, model_name: str, gpu_vram_gb: float = 12.0
    ) -> str:
        """Recommend quantization method based on model size and GPU VRAM."""
        config = self.get_model_config(model_name)
        size = config.get("size", "8B")

        rec = QUANT_RECOMMENDATIONS.get(size, {"recommended_quant": "awq"})
        min_vram = rec.get("min_vram_gb", 10)

        if gpu_vram_gb >= min_vram * 2:
            return "none"
        elif gpu_vram_gb >= min_vram:
            return rec["recommended_quant"]
        else:
            return "awq"  # Most aggressive compression


class DockerImageManager:
    """Manages Docker image operations for vLLM."""

    @staticmethod
    def ensure_image(image: str) -> bool:
        """Ensure vLLM image is available, pull from mirror if needed."""
        result = subprocess.run(
            f"docker image inspect {image}",
            shell=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return True

        # Try direct pull first
        logger.mesg(f"[qvl] Pulling image: {image}")
        try:
            subprocess.run(f"docker pull {image}", shell=True, check=True)
            return True
        except subprocess.CalledProcessError:
            pass

        # Fallback: pull from mirror
        mirror_image = f"{VLLM_IMAGE_MIRROR}/{image}"
        logger.mesg(f"[qvl] Pulling image from mirror: {mirror_image}")
        try:
            subprocess.run(f"docker pull {mirror_image}", shell=True, check=True)
            subprocess.run(f"docker tag {mirror_image} {image}", shell=True, check=True)
            logger.okay(f"[qvl] Image tagged as: {image}")
            return True
        except subprocess.CalledProcessError as e:
            logger.warn(f"× Failed to pull image: {e}")
            return False


class ComposeFileGenerator:
    """Generates docker-compose.yml content for vLLM deployment."""

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
        quantization: Optional[str] = None,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int = MAX_NUM_SEQS,
        limit_mm_per_prompt: str = LIMIT_MM_PER_PROMPT,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
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
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.gpu_memory_utilization = gpu_memory_utilization
        self.gpu_configs = gpu_configs
        self._gpu_config_map: dict[int, GpuModelConfig] = {}
        if gpu_configs:
            for gc in gpu_configs:
                self._gpu_config_map[gc.gpu_id] = gc

    def _get_gpu_config(self, gpu: GPUInfo) -> GpuModelConfig:
        """Get model config for a specific GPU."""
        if gpu.index in self._gpu_config_map:
            return self._gpu_config_map[gpu.index]
        quant_method = self.quantization or DEFAULT_QUANT_METHOD
        quant_level = DEFAULT_QUANT_LEVEL if quant_method == "awq" else ""
        return GpuModelConfig(
            gpu_id=gpu.index,
            model_name=self.model_name,
            quant_method=quant_method,
            quant_level=quant_level,
        )

    def generate(self) -> str:
        """Generate docker-compose.yml content."""
        lines = self._generate_header()
        lines.extend(self._generate_common_config())
        lines.append("services:")
        for i, gpu in enumerate(self.gpus):
            service_lines = self._generate_service(gpu=gpu)
            lines.extend(service_lines)
        return "\n".join(lines)

    def _generate_header(self) -> list[str]:
        """Generate compose file header."""
        lines = [
            f"# QVL (Qwen3-VL) Multi-GPU Deployment via vLLM",
        ]
        if self._gpu_config_map:
            lines.append(f"# Per-GPU configs:")
            for gpu in self.gpus:
                gc = self._get_gpu_config(gpu)
                lines.append(f"#   GPU {gc.gpu_id}: {gc.label}")
        else:
            quant_info = (
                f", quantization: {self.quantization}" if self.quantization else ""
            )
            lines.append(f"# Model: {self.model_name}{quant_info}")
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
                f"    - VLLM_WORKER_MULTIPROC_METHOD=spawn",
            ]
        )

        if self.http_proxy:
            lines.extend(
                [
                    f"    - HTTP_PROXY={self.http_proxy}",
                    f"    - HTTPS_PROXY={self.http_proxy}",
                    f"    - http_proxy={self.http_proxy}",
                    f"    - https_proxy={self.http_proxy}",
                    f"    - NO_PROXY=localhost,127.0.0.1",
                    f"    - no_proxy=localhost,127.0.0.1",
                ]
            )

        if self.mount_mode == "manual":
            lines.append(f"    - LD_LIBRARY_PATH=/usr/local/nvidia/lib64")
            # CUDA_VISIBLE_DEVICES=0 because each container mounts only one GPU
            lines.append(f"    - CUDA_VISIBLE_DEVICES=0")

        # vLLM requires ipc: host for shared memory
        lines.extend(
            [
                f"  ipc: host",
                f"",
            ]
        )
        return lines

    def _generate_service(self, gpu: GPUInfo) -> list[str]:
        """Generate service definition for a single GPU."""
        gpu_config = self._get_gpu_config(gpu)
        service_port = self.port + gpu.index
        container_name = f"{self.project_name}--gpu{gpu.index}"

        lines = [
            f"  qvl-gpu{gpu.index}:",
            f"    <<: *common-config",
            f"    image: {gpu.image}",
            f"    container_name: {container_name}",
        ]

        # Network mode
        if self.http_proxy:
            lines.append(f"    network_mode: host")
        else:
            lines.extend(
                [
                    f"    ports:",
                    f'      - "{service_port}:{VLLM_INTERNAL_PORT}"',
                ]
            )

        # Device mounting
        if self.mount_mode == "manual":
            lines.append(f"    devices:")
            lines.append(f"      - /dev/nvidia{gpu.index}:/dev/nvidia{gpu.index}")
            for device in NvidiaDriverLibs.get_required_devices():
                if Path(device).exists():
                    lines.append(f"      - {device}:{device}")
        else:
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

        # vLLM command arguments
        # Note: newer vLLM (>=0.12) uses positional model arg, not --model flag
        vllm_model = gpu_config.vllm_model_arg
        lines.extend(
            [
                f"    command:",
                f"      - {vllm_model}",
            ]
        )

        lines.extend(
            [
                f"      - --max-model-len",
                f'      - "{self.max_model_len}"',
                f"      - --max-num-seqs",
                f'      - "{self.max_num_seqs}"',
                f"      - --limit-mm-per-prompt",
                f"      - '{self.limit_mm_per_prompt}'",
                f"      - --gpu-memory-utilization",
                f'      - "{gpu_config.gpu_memory_utilization}"',
                f"      - --dtype",
                f"      - half",
                f"      - --trust-remote-code",
            ]
        )

        # Quantization (per-GPU config takes priority)
        quant = gpu_config.quant_method
        if quant and quant != "none":
            if quant == "bitsandbytes":
                lines.extend(
                    [
                        f"      - --quantization",
                        f"      - bitsandbytes",
                        f"      - --load-format",
                        f"      - bitsandbytes",
                    ]
                )
            # AWQ (cyankiwi models): vLLM auto-detects from model config.json
            # (quant_method=compressed-tensors), no --quantization flag needed.

        # HuggingFace token
        if self.hf_token:
            lines.extend(
                [
                    f"      - --api-key",
                    f"      - {self.hf_token}",
                ]
            )

        # Port configuration
        if self.http_proxy:
            lines.extend(
                [
                    f"      - --port",
                    f'      - "{service_port}"',
                ]
            )

        # Healthcheck
        lines.append(f"    healthcheck:")
        if self.http_proxy:
            lines.append(
                f'      test: ["CMD", "curl", "-f", "http://localhost:{service_port}/health"]'
            )
        else:
            lines.append(
                f'      test: ["CMD", "curl", "-f", "http://localhost:{VLLM_INTERNAL_PORT}/health"]'
            )

        lines.extend(
            [
                f"      interval: 30s",
                f"      timeout: 10s",
                f"      retries: 3",
                f"      start_period: 120s",  # vLLM takes longer to start than TEI
            ]
        )

        lines.append(f"")
        return lines


class QVLComposer:
    """Composer for QVL Docker Compose deployments."""

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
        quantization: Optional[str] = None,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int = MAX_NUM_SEQS,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        gpu_configs: list[GpuModelConfig] | None = None,
    ):
        self.model_name = model_name
        self.port = port
        self.gpu_ids = gpu_ids
        self.hf_token = hf_token
        self.mount_mode = mount_mode
        self.http_proxy = http_proxy
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.gpu_configs = gpu_configs

        # project name: lowercase, safe characters
        if gpu_configs:
            self.project_name = project_name or "qvl-multi"
        else:
            project_dash = model_name.replace("/", "--").lower()
            project_dash = re.sub(r"[^a-z0-9_-]", "_", project_dash)
            self.project_name = project_name or f"qvl--{project_dash}"

        # Compose file location
        if compose_dir:
            self.compose_dir = Path(compose_dir)
        else:
            script_dir = Path(__file__).resolve().parent
            self.compose_dir = script_dir.parent / "configs"

        self.compose_file = self.compose_dir / f"{self.project_name}.yml"

        # Components
        self.gpus = GPUDetector.detect(gpu_ids)
        self.model_config_manager = ModelConfigManager()
        self.image_manager = DockerImageManager()

    def _find_compose_file(self) -> Path | None:
        """Find an existing QVL compose file.

        Tries in order:
        1. The configured compose file path
        2. qvl-multi.yml (common multi-GPU deploy)
        3. Any qvl*.yml file in the configs directory
        """
        if self.compose_file.exists():
            return self.compose_file

        multi = self.compose_dir / "qvl-multi.yml"
        if multi.exists():
            return multi

        # Search for any qvl compose file
        candidates = sorted(self.compose_dir.glob("qvl*.yml"))
        if candidates:
            return candidates[0]

        return None

    @staticmethod
    def _find_qvl_containers(include_stopped: bool = False) -> list[str]:
        """Find all QVL-related Docker containers.

        Matches containers whose names start with 'qvl-' or 'qvl--'.
        """
        flag = "-a" if include_stopped else ""
        result = subprocess.run(
            f'docker ps {flag} --filter "name=qvl" ' f'--format "{{{{.Names}}}}"',
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        names = [
            n.strip()
            for n in result.stdout.strip().split("\n")
            if n.strip() and re.match(r"qvl[-_]", n.strip())
        ]
        return sorted(names)

    def _resolve_compose_cmd(self, cmd: str, require_file: bool = True) -> bool:
        """Try to run a compose command, auto-discovering the compose file.

        Returns True if the command was executed, False otherwise.
        """
        compose_file = self._find_compose_file()
        if compose_file:
            if compose_file != self.compose_file:
                logger.mesg(f"[qvl] Using compose file: {compose_file}")
            full_cmd = f"docker compose -f {compose_file} {cmd}"
            logger.mesg(f"[qvl] Running: {full_cmd}")
            subprocess.run(full_cmd, shell=True)
            return True
        return False

    def _get_service_name(self, gpu: GPUInfo) -> str:
        return f"qvl-gpu{gpu.index}"

    def _get_container_name(self, gpu: GPUInfo) -> str:
        return f"{self.project_name}--gpu{gpu.index}"

    def _ensure_compose_dir(self) -> None:
        self.compose_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_data_dir(self) -> Path:
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
            quantization=self.quantization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            gpu_configs=self.gpu_configs,
        )
        content = compose_generator.generate()
        self.compose_file.write_text(content)
        logger.okay(f"[qvl] Generated: {self.compose_file}")
        return self.compose_file

    def _run_compose_cmd(
        self, cmd: str, capture_output: bool = False
    ) -> subprocess.CompletedProcess:
        full_cmd = f"docker compose -f {self.compose_file} {cmd}"
        logger.mesg(f"[qvl] Running: {full_cmd}")
        return subprocess.run(full_cmd, shell=True, capture_output=capture_output)

    def up(self) -> None:
        """Start all QVL containers."""
        if not self.gpus:
            logger.warn("× No healthy GPUs detected")
            gpu_summary = GPUDetector.get_unhealthy_gpu_summary()
            if gpu_summary:
                logger.warn(f"[qvl] GPU issues detected:")
                for line in gpu_summary.split("\n"):
                    if line.strip():
                        logger.warn(f"  {line.strip()}")
            return

        logger.mesg(f"[qvl] Starting vLLM for model: {self.model_name}")
        if self.gpu_configs:
            for gc in self.gpu_configs:
                logger.mesg(f"[qvl]   GPU {gc.gpu_id}: {gc.label}")
        logger.mesg(
            f"[qvl] GPUs: {[f'{g.index}(cap={g.compute_cap})' for g in self.gpus]}"
        )
        if self.quantization:
            logger.mesg(f"[qvl] Quantization: {self.quantization}")

        # Ensure image
        self.image_manager.ensure_image(VLLM_IMAGE)

        # Ensure directories
        self._ensure_data_dir()

        # Generate compose file
        self.generate_compose_file()

        # Start services
        logger.mesg(f"[qvl] Starting all services...")
        try:
            result = self._run_compose_cmd(
                "up -d --remove-orphans", capture_output=True
            )
            if result.returncode != 0:
                error_msg = ""
                if result.stderr:
                    if isinstance(result.stderr, bytes):
                        error_msg = result.stderr.decode("utf-8", errors="replace")
                    else:
                        error_msg = result.stderr
                logger.warn(f"[qvl] Startup failed: {error_msg.strip()}")
            else:
                logger.okay(
                    f"[qvl] All GPU services started: {[g.index for g in self.gpus]}"
                )
        except Exception as e:
            logger.warn(f"[qvl] Startup error: {e}")

        self.ps()

    def down(self) -> None:
        """Stop and remove all QVL containers."""
        if self._resolve_compose_cmd("down --remove-orphans"):
            return

        # Fallback: find and remove all QVL containers directly
        container_names = self._find_qvl_containers(include_stopped=True)
        if not container_names:
            logger.mesg("[qvl] No QVL containers found")
            return

        logger.mesg(
            f"[qvl] Removing {len(container_names)} container(s): {', '.join(container_names)}"
        )
        for name in container_names:
            subprocess.run(f"docker rm -f {name}", shell=True, capture_output=True)
        logger.okay(f"[qvl] Removed {len(container_names)} container(s)")

    def stop(self) -> None:
        """Stop all QVL containers."""
        if self._resolve_compose_cmd("stop"):
            return

        # Fallback: stop all running QVL containers directly
        container_names = self._find_qvl_containers(include_stopped=False)
        if not container_names:
            logger.mesg("[qvl] No running QVL containers found")
            return

        logger.mesg(
            f"[qvl] Stopping {len(container_names)} container(s): {', '.join(container_names)}"
        )
        for name in container_names:
            subprocess.run(f"docker stop {name}", shell=True, capture_output=True)
        logger.okay(f"[qvl] Stopped {len(container_names)} container(s)")

    def start(self) -> None:
        """Start stopped QVL containers."""
        if self._resolve_compose_cmd("start"):
            return

        # Fallback: start all stopped QVL containers
        container_names = self._find_qvl_containers(include_stopped=True)
        running = set(self._find_qvl_containers(include_stopped=False))
        stopped = [n for n in container_names if n not in running]
        if not stopped:
            logger.mesg("[qvl] No stopped QVL containers found")
            return

        logger.mesg(f"[qvl] Starting {len(stopped)} container(s): {', '.join(stopped)}")
        for name in stopped:
            subprocess.run(f"docker start {name}", shell=True, capture_output=True)
        logger.okay(f"[qvl] Started {len(stopped)} container(s)")

    def restart(self) -> None:
        """Restart all QVL containers."""
        if self._resolve_compose_cmd("restart"):
            return

        # Fallback: restart all QVL containers directly
        container_names = self._find_qvl_containers(include_stopped=False)
        if not container_names:
            logger.mesg("[qvl] No running QVL containers found")
            return

        logger.mesg(
            f"[qvl] Restarting {len(container_names)} container(s): {', '.join(container_names)}"
        )
        for name in container_names:
            subprocess.run(f"docker restart {name}", shell=True, capture_output=True)
        logger.okay(f"[qvl] Restarted {len(container_names)} container(s)")

    def ps(self) -> None:
        if self._resolve_compose_cmd("ps"):
            return
        self._show_manual_status()

    def logs(self, follow: bool = False, tail: int = 100) -> None:
        follow_flag = "-f" if follow else ""
        cmd = f"logs --tail={tail} {follow_flag}".strip()
        if self._resolve_compose_cmd(cmd):
            return

        # Fallback: show logs for individual containers
        container_names = self._find_qvl_containers(include_stopped=True)
        if not container_names:
            logger.mesg("[qvl] No QVL containers found")
            return

        for name in container_names:
            logger.mesg(f"[qvl] Logs for {name}:")
            follow_arg = "-f" if follow else ""
            subprocess.run(
                f"docker logs --tail={tail} {follow_arg} {name}".strip(),
                shell=True,
            )

    def _show_manual_status(self) -> None:
        """Show status by querying Docker directly."""
        logger.mesg(f"[qvl] Status for: {self.model_name}")
        print("=" * 85)
        print(f"{'GPU':<5} {'CONTAINER':<42} {'PORT':<7} {'STATUS':<12} {'HEALTH':<10}")
        print("-" * 85)

        for gpu in self.gpus:
            container_name = self._get_container_name(gpu)
            container_port = self.port + gpu.index

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

            is_healthy, _ = GPUDetector.check_gpu_health(gpu.index)
            gpu_health = "healthy" if is_healthy else "unhealthy"

            print(
                f"{gpu.index:<5} {container_name:<42} {container_port:<7} {container_status:<12} {gpu_health:<10}"
            )

        print("=" * 85)

    def health(self) -> None:
        """Check GPU health status."""
        logger.mesg(f"[qvl] GPU Health Check")
        print("=" * 70)
        print(f"{'GPU':<6} {'STATUS':<12} {'INFO':<50}")
        print("-" * 70)

        for gpu in self.gpus:
            is_healthy, message = GPUDetector.check_gpu_health(gpu.index)
            status = "healthy" if is_healthy else "unhealthy"
            info = message[:48] + ".." if len(message) > 50 else message
            print(f"{gpu.index:<6} {status:<12} {info:<50}")

        print("=" * 70)


class QVLComposeArgParser:
    """Argument parser for QVL Compose CLI."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="QVL Docker Compose Manager",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _add_common_arguments(self, parser):
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
            help="Project name (default: qvl--MODEL_NAME)",
        )
        parser.add_argument(
            "-g",
            "--gpus",
            type=str,
            default=None,
            help="Comma-separated GPU IDs (default: all)",
        )

    def _add_deployment_arguments(self, parser):
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
            help="HTTP/HTTPS proxy for model downloads",
        )
        parser.add_argument(
            "-q",
            "--quantization",
            type=str,
            choices=["none", "awq", "bitsandbytes"],
            default=None,
            help="Quantization method (default: awq)",
        )
        parser.add_argument(
            "--max-model-len",
            type=int,
            default=MAX_MODEL_LEN,
            help=f"Max model context length (default: {MAX_MODEL_LEN})",
        )
        parser.add_argument(
            "--max-num-seqs",
            type=int,
            default=MAX_NUM_SEQS,
            help=f"Max concurrent sequences (default: {MAX_NUM_SEQS})",
        )
        parser.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=GPU_MEMORY_UTILIZATION,
            help=f"GPU memory utilization ratio (default: {GPU_MEMORY_UTILIZATION})",
        )
        parser.add_argument(
            "--gpu-configs",
            type=str,
            default=None,
            help=(
                "Per-GPU model/quant configs. "
                'Format: "GPU:MODEL:QUANT,...". '
                'Example: "0:2B-Instruct:4bit,1:8B-Instruct:4bit"'
            ),
        )

    def _setup_arguments(self):
        subparsers = self.parser.add_subparsers(
            dest="action",
            help="Action to perform",
            required=False,
        )

        parser_up = subparsers.add_parser("up", help="Start QVL containers")
        self._add_common_arguments(parser_up)
        self._add_deployment_arguments(parser_up)

        parser_down = subparsers.add_parser("down", help="Stop and remove containers")
        self._add_common_arguments(parser_down)

        parser_stop = subparsers.add_parser("stop", help="Stop containers")
        self._add_common_arguments(parser_stop)

        parser_start = subparsers.add_parser("start", help="Start stopped containers")
        self._add_common_arguments(parser_start)

        parser_restart = subparsers.add_parser("restart", help="Restart containers")
        self._add_common_arguments(parser_restart)

        parser_ps = subparsers.add_parser("ps", help="Show container status")
        self._add_common_arguments(parser_ps)

        parser_logs = subparsers.add_parser("logs", help="View logs")
        self._add_common_arguments(parser_logs)
        parser_logs.add_argument("-f", "--follow", action="store_true")
        parser_logs.add_argument("--tail", type=int, default=100)

        parser_gen = subparsers.add_parser("generate", help="Generate compose file")
        self._add_common_arguments(parser_gen)
        self._add_deployment_arguments(parser_gen)

        parser_health = subparsers.add_parser("health", help="Check GPU health")
        self._add_common_arguments(parser_health)


def main():
    arg_parser = QVLComposeArgParser()
    args = arg_parser.args

    if not args.action:
        arg_parser.parser.print_help()
        return

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
        "quantization": getattr(args, "quantization", None),
        "max_model_len": getattr(args, "max_model_len", MAX_MODEL_LEN),
        "max_num_seqs": getattr(args, "max_num_seqs", MAX_NUM_SEQS),
        "gpu_memory_utilization": getattr(
            args, "gpu_memory_utilization", GPU_MEMORY_UTILIZATION
        ),
    }

    # Parse per-GPU configs if provided
    gpu_configs_str = getattr(args, "gpu_configs", None)
    if gpu_configs_str:
        composer_kwargs["gpu_configs"] = parse_gpu_configs(gpu_configs_str)

    composer = QVLComposer(**composer_kwargs)

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
        composer.logs(
            follow=getattr(args, "follow", False),
            tail=getattr(args, "tail", 100),
        )
    elif args.action == "generate":
        composer.generate_compose_file()
    elif args.action == "health":
        composer.health()


if __name__ == "__main__":
    main()
