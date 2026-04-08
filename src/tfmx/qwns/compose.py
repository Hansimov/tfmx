"""QWN Docker Compose manager.

Deploys quantized Qwen 3.5 text models with vLLM, one container per GPU,
and provides Docker lifecycle helpers used by the unified ``qwn`` CLI.
"""

import argparse
import json
import re
import shlex
import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from tclogger import logger

from ..utils.service_bootstrap import wait_for_healthy_docker_containers

from .networking import DEFAULT_HF_ENDPOINT
from .networking import DEFAULT_PIP_INDEX_URL
from .networking import DEFAULT_PIP_TRUSTED_HOST
from .networking import QWNNetworkConfig


SERVER_PORT = 27880
MACHINE_PORT = 27800

MODEL_NAME = "Qwen/Qwen3.5-4B"
DEFAULT_AWQ_MODEL = "cyankiwi/Qwen3.5-4B-AWQ-4bit"

HF_ENDPOINT = DEFAULT_HF_ENDPOINT
CACHE_HF = ".cache/huggingface"
CACHE_HF_HUB = f"{CACHE_HF}/hub"
CACHE_VLLM = ".cache/vllm"

QWEN35_VLLM_VERSION = "0.19.0"
VLLM_BASE_IMAGE = f"vllm/vllm-openai:v{QWEN35_VLLM_VERSION}"
VLLM_IMAGE = "tfmx-vllm-openai:qwen3.5-v0.19.0"
VLLM_IMAGE_MIRROR = "m.daocloud.io"
VLLM_INTERNAL_PORT = 8000
QWEN35_TRANSFORMERS_SPEC = (
    "https://codeload.github.com/huggingface/transformers/tar.gz/refs/heads/main"
)
QWEN35_HF_HUB_SPEC = "huggingface-hub>=1.5.0,<2.0"
QWEN35_DOCKERFILE = "Dockerfile.qwen3.5-vllm"

MAX_CONCURRENT_REQUESTS = 8
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 8
GPU_MEMORY_UTILIZATION = 0.72
DEVICE_MOUNT_MODE = "manual"
GPU_LAYOUT_UNIFORM_AWQ = "uniform-awq"
DEFAULT_SKIP_MM_PROFILING = True
DEFAULT_CUDAGRAPH_MODE: str | None = None
DEFAULT_ENABLE_SLEEP_MODE = False
DEFAULT_SLEEP_LEVEL = 1
DEFAULT_SLEEP_MODE = "abort"
SLEEP_CONTROL_TIMEOUT_SEC = 5.0
SLEEP_WAKE_TIMEOUT_SEC = 180.0
SLEEP_WAKE_POLL_INTERVAL_SEC = 2.0
WARMUP_REQUEST_TIMEOUT_SEC = 60.0
WARMUP_HEALTH_TIMEOUT_SEC = 300.0
WARMUP_HEALTH_POLL_INTERVAL_SEC = 2.0
WARMUP_PROMPT = "你好，请只回答一个字：好"
WARMUP_MAX_TOKENS = 16

CUDAGRAPH_MODE_CHOICES = (
    "NONE",
    "PIECEWISE",
    "FULL",
    "FULL_DECODE_ONLY",
    "FULL_AND_PIECEWISE",
)

HEALTHCHECK_INTERVAL = "5s"
HEALTHCHECK_TIMEOUT = "3s"
HEALTHCHECK_RETRIES = 12
HEALTHCHECK_START_PERIOD = "240s"
HEALTHCHECK_TCP_PROBE = f"bash -lc 'exec 3<>/dev/tcp/127.0.0.1/{VLLM_INTERNAL_PORT}'"

SLEEP_STATE_FILE = Path.home() / ".cache" / "tfmx" / "qwn_sleep_state.json"

SUPPORTED_MODELS = {
    MODEL_NAME: {
        "size": "4B",
        "family": "qwen3.5",
        "type": "chat",
        "max_model_len": MAX_MODEL_LEN,
    }
}

AWQ_MODELS = {
    DEFAULT_AWQ_MODEL: MODEL_NAME,
}

MODEL_SHORTCUTS = {
    "4b": MODEL_NAME,
    "qwen3.5-4b": MODEL_NAME,
    "qwen-4b": MODEL_NAME,
}

MODEL_ALIASES = {
    "4b-awq": "4b",
    "qwen3.5": "4b",
    "qwen3.5-4b-awq": "4b",
    "qwen3.5-4b-awq-4bit": "4b",
    "default": "4b",
}

MODEL_SHORTCUT_REV = {MODEL_NAME: "4b"}
MODEL_SHORTCUT_REV_LOWER = {MODEL_NAME.lower(): "4b"}

DISPLAY_SHORTCUTS = {
    "4b": "4B",
}

DEFAULT_QUANT_METHOD = "awq"
DEFAULT_QUANT_LEVEL = "4bit"
AWQ_QUANT_LEVELS = {"4bit"}

AWQ_REPO_MAP = {
    ("4b", "4bit"): DEFAULT_AWQ_MODEL,
}

QUANT_RECOMMENDATIONS = {
    "4B": {
        "min_vram_gb": 8,
        "recommended_quant": "awq",
    }
}

GPU_MEMORY_UTILIZATION_BY_SIZE = {
    "4B": GPU_MEMORY_UTILIZATION,
}

GPU_COMPUTE_CAPS = {
    "8.6": "RTX 30xx",
    "8.9": "RTX 40xx",
    "8.0": "A100/A30",
    "9.0": "H100",
}


def normalize_model_key(key: str) -> str:
    return key.strip().lower() if key else ""


def _split_quant_suffix(value: str) -> tuple[str, str]:
    normalized = normalize_model_key(value)
    if ":" not in normalized:
        return normalized, ""
    model_part, maybe_quant = normalized.rsplit(":", 1)
    if maybe_quant in AWQ_QUANT_LEVELS:
        return model_part, maybe_quant
    return normalized, ""


def resolve_model_name(key: str) -> str:
    if not key:
        return MODEL_NAME

    model_key, _ = _split_quant_suffix(key)
    if model_key in MODEL_ALIASES:
        model_key = MODEL_ALIASES[model_key]

    if model_key in MODEL_SHORTCUTS:
        return MODEL_SHORTCUTS[model_key]

    for full_name in SUPPORTED_MODELS:
        if full_name.lower() == model_key:
            return full_name

    for awq_name, base_name in AWQ_MODELS.items():
        if awq_name.lower() == model_key:
            return base_name

    return key


def resolve_quant_level(level: str) -> str:
    return normalize_model_key(level)


def get_model_shortcut(model_name: str) -> str:
    if not model_name:
        return ""

    stripped_model, quant = _split_quant_suffix(model_name)
    if stripped_model in MODEL_ALIASES:
        stripped_model = MODEL_ALIASES[stripped_model]

    direct = MODEL_SHORTCUT_REV.get(model_name)
    if direct:
        return direct

    lower_direct = MODEL_SHORTCUT_REV_LOWER.get(stripped_model)
    if lower_direct:
        return lower_direct

    if stripped_model in MODEL_SHORTCUTS:
        return stripped_model

    for awq_name, base_name in AWQ_MODELS.items():
        if awq_name.lower() == stripped_model:
            return MODEL_SHORTCUT_REV.get(base_name, "4b")

    if stripped_model in MODEL_ALIASES:
        return MODEL_ALIASES[stripped_model]

    tail = stripped_model.split("/")[-1]
    match = re.search(r"(\d+b)", tail)
    if match:
        return match.group(1)
    return tail or model_name


def get_display_shortcut(shortcut: str) -> str:
    normalized = normalize_model_key(shortcut)
    if normalized in MODEL_ALIASES:
        normalized = MODEL_ALIASES[normalized]
    return DISPLAY_SHORTCUTS.get(normalized, shortcut)


def get_model_api_aliases(model_name: str, quant_level: str = "") -> list[str]:
    shortcut = normalize_model_key(get_model_shortcut(model_name))
    quant = normalize_model_key(quant_level)

    aliases: list[str] = []
    if shortcut == "4b":
        if quant == "4bit":
            aliases.append("qwen3.5-4b-awq-4bit")
        aliases.append("qwen3.5-4b")

    deduped: list[str] = []
    for alias in aliases:
        if alias and alias not in deduped:
            deduped.append(alias)
    return deduped


def _normalize_backend_endpoint(endpoint: str) -> str:
    return endpoint.rstrip("/") if endpoint else ""


def load_backend_sleep_states() -> dict[str, bool]:
    if not SLEEP_STATE_FILE.exists():
        return {}

    try:
        payload = json.loads(SLEEP_STATE_FILE.read_text())
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    states: dict[str, bool] = {}
    for endpoint, sleeping in payload.items():
        normalized = _normalize_backend_endpoint(str(endpoint))
        if normalized:
            states[normalized] = bool(sleeping)
    return states


def _write_backend_sleep_states(states: dict[str, bool]) -> None:
    if not states:
        try:
            SLEEP_STATE_FILE.unlink(missing_ok=True)
        except OSError:
            pass
        return

    SLEEP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SLEEP_STATE_FILE.write_text(json.dumps(states, indent=2, sort_keys=True) + "\n")


def update_backend_sleep_states(state_updates: dict[str, bool]) -> None:
    normalized_updates = {
        _normalize_backend_endpoint(endpoint): bool(sleeping)
        for endpoint, sleeping in state_updates.items()
        if _normalize_backend_endpoint(endpoint)
    }
    if not normalized_updates:
        return

    states = load_backend_sleep_states()
    states.update(normalized_updates)
    _write_backend_sleep_states(states)


def set_backend_sleep_states(endpoints: list[str], *, sleeping: bool) -> None:
    update_backend_sleep_states({endpoint: sleeping for endpoint in endpoints})


def clear_backend_sleep_states(endpoints: list[str]) -> None:
    normalized_endpoints = {
        _normalize_backend_endpoint(endpoint) for endpoint in endpoints if endpoint
    }
    if not normalized_endpoints:
        return

    states = load_backend_sleep_states()
    changed = False
    for endpoint in normalized_endpoints:
        if endpoint in states:
            del states[endpoint]
            changed = True
    if changed:
        _write_backend_sleep_states(states)


def get_backend_sleep_state(endpoint: str) -> bool | None:
    normalized = _normalize_backend_endpoint(endpoint)
    if not normalized:
        return None
    return load_backend_sleep_states().get(normalized)


@dataclass
class GpuModelConfig:
    gpu_id: int
    model_name: str = MODEL_NAME
    quant_method: str = DEFAULT_QUANT_METHOD
    quant_level: str = DEFAULT_QUANT_LEVEL

    def __post_init__(self) -> None:
        self.model_name = resolve_model_name(self.model_name)
        self.quant_method = (
            normalize_model_key(self.quant_method) or DEFAULT_QUANT_METHOD
        )
        self.quant_level = resolve_quant_level(self.quant_level) or DEFAULT_QUANT_LEVEL

    @property
    def model_shortcut(self) -> str:
        return get_model_shortcut(self.model_name)

    @property
    def display_shortcut(self) -> str:
        return get_display_shortcut(self.model_shortcut)

    @property
    def model_size(self) -> str:
        config = SUPPORTED_MODELS.get(self.model_name)
        if config:
            return config["size"]
        return "4B"

    @property
    def gpu_memory_utilization(self) -> float:
        return GPU_MEMORY_UTILIZATION_BY_SIZE.get(
            self.model_size,
            GPU_MEMORY_UTILIZATION,
        )

    @property
    def awq_repo(self) -> str | None:
        if self.quant_method != "awq":
            return None
        return AWQ_REPO_MAP.get((self.model_shortcut, self.quant_level))

    @property
    def vllm_model_arg(self) -> str:
        if self.quant_method == "awq":
            repo = self.awq_repo
            if repo:
                return repo
        return self.model_name

    @property
    def label(self) -> str:
        if self.quant_level:
            return f"{self.model_shortcut}:{self.quant_level}"
        return self.model_shortcut

    @property
    def display_label(self) -> str:
        if self.quant_level:
            return f"{self.display_shortcut}:{self.quant_level}"
        return self.display_shortcut

    @property
    def served_model_name(self) -> str:
        return self.label

    def to_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "model_name": self.model_name,
            "model_shortcut": self.model_shortcut,
            "quant_method": self.quant_method,
            "quant_level": self.quant_level,
            "awq_repo": self.awq_repo,
            "served_model_name": self.served_model_name,
        }


def parse_gpu_configs(config_str: str) -> list[GpuModelConfig]:
    configs: list[GpuModelConfig] = []
    seen_gpu_ids: set[int] = set()
    for raw_part in config_str.split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = [field.strip() for field in part.split(":")]
        if not fields or not fields[0]:
            raise ValueError(
                f"Invalid config: '{part}'. Format: GPU_ID[:MODEL[:QUANT]]"
            )
        if len(fields) > 3:
            raise ValueError(
                f"Invalid config: '{part}'. Format: GPU_ID[:MODEL[:QUANT]]"
            )

        gpu_id = int(fields[0])
        if gpu_id in seen_gpu_ids:
            raise ValueError(f"Duplicate GPU ID in config: '{gpu_id}'")
        seen_gpu_ids.add(gpu_id)
        model_name = (
            resolve_model_name(fields[1])
            if len(fields) > 1 and fields[1]
            else MODEL_NAME
        )
        quant_level = (
            resolve_quant_level(fields[2])
            if len(fields) > 2 and fields[2]
            else DEFAULT_QUANT_LEVEL
        )
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


def parse_cudagraph_capture_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    seen: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise ValueError("cudagraph capture sizes must be positive integers")
        if size in seen:
            continue
        sizes.append(size)
        seen.add(size)
    if not sizes:
        raise ValueError("expected at least one cudagraph capture size")
    return sizes


def build_cudagraph_capture_sizes(max_num_seqs: int) -> list[int]:
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")

    sizes: list[int] = []
    size = 1
    while size < max_num_seqs:
        sizes.append(size)
        size *= 2

    if not sizes or sizes[-1] != max_num_seqs:
        sizes.append(max_num_seqs)
    return sizes


def build_compilation_config(
    cudagraph_mode: str | None = DEFAULT_CUDAGRAPH_MODE,
    max_num_seqs: int = MAX_NUM_SEQS,
    cudagraph_capture_sizes: list[int] | None = None,
) -> dict[str, object] | None:
    if not cudagraph_mode:
        return None

    mode = cudagraph_mode.upper()
    if mode not in CUDAGRAPH_MODE_CHOICES:
        raise ValueError(f"Unsupported cudagraph mode: {mode}")

    compilation_config: dict[str, object] = {"cudagraph_mode": mode}
    if mode == "NONE":
        return compilation_config

    capture_sizes = cudagraph_capture_sizes or build_cudagraph_capture_sizes(
        max_num_seqs
    )
    compilation_config["cudagraph_capture_sizes"] = capture_sizes
    compilation_config["max_cudagraph_capture_size"] = max(capture_sizes)
    return compilation_config


def build_gpu_configs_for_layout(
    layout: str,
    gpus: list["GPUInfo"],
) -> list[GpuModelConfig]:
    normalized_layout = normalize_model_key(layout)
    if normalized_layout != GPU_LAYOUT_UNIFORM_AWQ:
        raise ValueError(f"Unsupported GPU layout: {layout}")

    return [
        GpuModelConfig(
            gpu_id=gpu.index,
            model_name=MODEL_NAME,
            quant_method=DEFAULT_QUANT_METHOD,
            quant_level=DEFAULT_QUANT_LEVEL,
        )
        for gpu in gpus
    ]


class NvidiaDriverLibs:
    @staticmethod
    def detect_driver_lib_dir() -> Optional[str]:
        candidates = [
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/local/nvidia/lib64",
        ]
        for path in candidates:
            if Path(path).exists() and list(Path(path).glob("libcuda.so*")):
                return path

        try:
            result = subprocess.run(
                "ldconfig -p | grep 'libcuda.so\\.' | grep x86-64 | head -1 | awk '{print $NF}'",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return str(Path(result.stdout.strip()).parent)
        except Exception:
            pass
        return None

    @staticmethod
    def get_required_devices() -> list[str]:
        return [
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            "/dev/nvidia-modeset",
        ]


class GPUInfo:
    def __init__(self, index: int, compute_cap: str):
        self.index = index
        self.compute_cap = compute_cap
        self.arch_name = GPU_COMPUTE_CAPS.get(compute_cap, "Unknown")
        self.image = VLLM_IMAGE

    def __repr__(self) -> str:
        return f"GPU({self.index}, cap={self.compute_cap}, arch={self.arch_name})"


class GPUDetector:
    @staticmethod
    def detect(
        gpu_ids: Optional[str] = None,
        check_health: bool = True,
    ) -> list[GPUInfo]:
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
            )

            gpus: list[GPUInfo] = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    gpus.append(GPUInfo(index=int(parts[0]), compute_cap=parts[1]))
                except ValueError:
                    continue

            if gpu_ids:
                specified = {
                    int(value.strip()) for value in gpu_ids.split(",") if value.strip()
                }
                gpus = [gpu for gpu in gpus if gpu.index in specified]

            if not check_health:
                return gpus

            healthy_gpus: list[GPUInfo] = []
            for gpu in gpus:
                healthy, message = GPUDetector.check_gpu_health(gpu.index)
                if healthy:
                    healthy_gpus.append(gpu)
                else:
                    logger.warn(
                        f"[qwn] GPU {gpu.index} excluded (unhealthy): {message}"
                    )
            return healthy_gpus
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warn(f"× Failed to detect GPUs: {exc}")
            return []

    @staticmethod
    def check_gpu_health(gpu_index: int) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                (
                    f"nvidia-smi -i {gpu_index} --query-gpu=name,memory.total,driver_version "
                    "--format=csv,noheader,nounits"
                ),
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False, result.stderr.strip() or "nvidia-smi failed"
            if not result.stdout.strip():
                return False, "No GPU info returned"
            return True, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, "nvidia-smi timeout"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def get_unhealthy_gpu_summary() -> str:
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.stderr and result.stderr.strip():
                return result.stderr.strip()
        except Exception:
            pass
        return ""


class ModelConfigManager:
    def __init__(self, cache_hf_hub: str = CACHE_HF_HUB):
        self.cache_hf_hub = cache_hf_hub

    def get_model_snapshot_dir(self, model_name: str) -> Optional[Path]:
        model_name_dash = model_name.replace("/", "--")
        cache_path = Path.home() / self.cache_hf_hub
        model_dir = cache_path / f"models--{model_name_dash}"
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                return snapshot
        return None

    def get_model_config(self, model_name: str) -> dict:
        if model_name in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_name]

        base_model = AWQ_MODELS.get(model_name)
        if base_model and base_model in SUPPORTED_MODELS:
            config = SUPPORTED_MODELS[base_model].copy()
            config["awq"] = True
            config["base_model"] = base_model
            return config

        return {
            "size": "4B",
            "family": "unknown",
            "type": "chat",
            "max_model_len": MAX_MODEL_LEN,
        }

    def get_quantization_recommendation(
        self,
        model_name: str,
        gpu_vram_gb: float = 12.0,
    ) -> str:
        config = self.get_model_config(model_name)
        size = config.get("size", "4B")
        recommendation = QUANT_RECOMMENDATIONS.get(size, {"recommended_quant": "awq"})
        min_vram = recommendation.get("min_vram_gb", 8)
        if gpu_vram_gb >= min_vram * 2:
            return "none"
        if gpu_vram_gb >= min_vram:
            return recommendation["recommended_quant"]
        return "awq"


class DockerImageManager:
    @staticmethod
    def _image_exists(image: str) -> bool:
        inspect_result = subprocess.run(
            f"docker image inspect {image}",
            shell=True,
            capture_output=True,
        )
        return inspect_result.returncode == 0

    @staticmethod
    def ensure_base_image(image: str = VLLM_BASE_IMAGE) -> bool:
        if DockerImageManager._image_exists(image):
            return True

        logger.mesg(f"[qwn] Pulling image: {image}")
        try:
            subprocess.run(f"docker pull {image}", shell=True, check=True)
            return True
        except subprocess.CalledProcessError:
            pass

        mirror_image = f"{VLLM_IMAGE_MIRROR}/{image}"
        logger.mesg(f"[qwn] Pulling image from mirror: {mirror_image}")
        try:
            subprocess.run(f"docker pull {mirror_image}", shell=True, check=True)
            subprocess.run(f"docker tag {mirror_image} {image}", shell=True, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            logger.warn(f"× Failed to pull image: {exc}")
            return False

    @staticmethod
    def _docker_data_dir() -> Path:
        return Path(__file__).resolve().parent / "docker_data"

    @staticmethod
    def ensure_qwen35_image(
        image: str = VLLM_IMAGE,
        network_config: QWNNetworkConfig | None = None,
    ) -> bool:
        network_config = network_config or QWNNetworkConfig.from_overrides()
        if DockerImageManager._image_exists(image):
            return True

        if not DockerImageManager.ensure_base_image(VLLM_BASE_IMAGE):
            return False

        dockerfile_path = DockerImageManager._docker_data_dir() / QWEN35_DOCKERFILE
        if not dockerfile_path.exists():
            logger.warn(f"× Missing runtime Dockerfile: {dockerfile_path}")
            return False

        logger.mesg(
            f"[qwn] Building runtime image: {image} (base={VLLM_BASE_IMAGE}, transformers={QWEN35_TRANSFORMERS_SPEC}, hub={QWEN35_HF_HUB_SPEC})"
        )
        build_cmd = [
            "docker",
            "build",
            "--progress=plain",
            "-t",
            image,
            "--build-arg",
            f"BASE_IMAGE={VLLM_BASE_IMAGE}",
            "--build-arg",
            f"TRANSFORMERS_SPEC={QWEN35_TRANSFORMERS_SPEC}",
            "--build-arg",
            f"HF_HUB_SPEC={QWEN35_HF_HUB_SPEC}",
        ]

        build_proxy = network_config.build_proxy
        if build_proxy:
            logger.mesg(f"[qwn] Docker build proxy: {build_proxy}")
        logger.mesg(f"[qwn] Pip mirror: {network_config.pip_index_url}")
        logger.mesg(f"[qwn] HF mirror: {network_config.hf_endpoint}")
        build_cmd.extend(network_config.docker_build_args())

        if network_config.use_host_network_for_build:
            logger.mesg("[qwn] Docker build uses host network for loopback proxy")
            build_cmd.extend(["--network", "host"])

        build_cmd.extend(
            [
                "-f",
                str(dockerfile_path),
                str(dockerfile_path.parent),
            ]
        )
        result = subprocess.run(build_cmd, text=True)
        if result.returncode != 0:
            logger.warn(
                f"× Failed to build runtime image: docker build exited with {result.returncode}"
            )
            return False

        logger.okay(f"[qwn] Built runtime image: {image}")
        return True

    @staticmethod
    def ensure_image(
        image: str = VLLM_IMAGE,
        network_config: QWNNetworkConfig | None = None,
    ) -> bool:
        if image == VLLM_IMAGE:
            return DockerImageManager.ensure_qwen35_image(
                image,
                network_config=network_config,
            )
        return DockerImageManager.ensure_base_image(image)


class ComposeFileGenerator:
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
        cache_vllm: str = CACHE_VLLM,
        hf_endpoint: str = HF_ENDPOINT,
        mount_mode: str = DEVICE_MOUNT_MODE,
        driver_lib_dir: Optional[str] = None,
        http_proxy: Optional[str] = None,
        pip_index_url: str = DEFAULT_PIP_INDEX_URL,
        pip_trusted_host: str = DEFAULT_PIP_TRUSTED_HOST,
        quantization: Optional[str] = None,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int = MAX_NUM_SEQS,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        skip_mm_profiling: bool = DEFAULT_SKIP_MM_PROFILING,
        enable_sleep_mode: bool = DEFAULT_ENABLE_SLEEP_MODE,
        cudagraph_mode: str | None = DEFAULT_CUDAGRAPH_MODE,
        cudagraph_capture_sizes: list[int] | None = None,
        gpu_configs: list[GpuModelConfig] | None = None,
        network_config: QWNNetworkConfig | None = None,
    ):
        self.gpus = gpus
        self.model_name = model_name
        self.port = port
        self.project_name = project_name
        self.data_dir = data_dir
        self.hf_token = hf_token
        self.cache_hf = cache_hf
        self.cache_hf_hub = cache_hf_hub
        self.cache_vllm = cache_vllm
        self.mount_mode = mount_mode
        self.driver_lib_dir = driver_lib_dir or NvidiaDriverLibs.detect_driver_lib_dir()
        self.network_config = network_config or QWNNetworkConfig.from_overrides(
            proxy=http_proxy,
            hf_endpoint=hf_endpoint,
            pip_index_url=pip_index_url,
            pip_trusted_host=pip_trusted_host,
        )
        self.quantization = quantization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.skip_mm_profiling = skip_mm_profiling
        self.enable_sleep_mode = enable_sleep_mode
        self.cudagraph_mode = cudagraph_mode.upper() if cudagraph_mode else None
        self.cudagraph_capture_sizes = cudagraph_capture_sizes
        self.compilation_config = build_compilation_config(
            cudagraph_mode=self.cudagraph_mode,
            max_num_seqs=self.max_num_seqs,
            cudagraph_capture_sizes=self.cudagraph_capture_sizes,
        )
        self.gpu_configs = gpu_configs or []
        self._gpu_config_map = {config.gpu_id: config for config in self.gpu_configs}
        self.model_config_manager = ModelConfigManager(cache_hf_hub=self.cache_hf_hub)

    def _get_gpu_config(self, gpu: GPUInfo) -> GpuModelConfig:
        if gpu.index in self._gpu_config_map:
            return self._gpu_config_map[gpu.index]

        quant_method = normalize_model_key(self.quantization) or DEFAULT_QUANT_METHOD
        quant_level = DEFAULT_QUANT_LEVEL if quant_method == "awq" else ""
        return GpuModelConfig(
            gpu_id=gpu.index,
            model_name=self.model_name,
            quant_method=quant_method,
            quant_level=quant_level,
        )

    def _uses_runtime_offline_cache(self) -> bool:
        model_ids = {self._get_gpu_config(gpu).vllm_model_arg for gpu in self.gpus}
        if not model_ids:
            return False

        for model_id in model_ids:
            snapshot_dir = self.model_config_manager.get_model_snapshot_dir(model_id)
            if snapshot_dir is None:
                return False
            if not any(
                (snapshot_dir / filename).exists()
                for filename in (
                    "config.json",
                    "preprocessor_config.json",
                    "processor_config.json",
                )
            ):
                return False
        return True

    def generate(self) -> str:
        lines = self._generate_header()
        lines.extend(self._generate_common_config())
        lines.append("services:")
        for gpu in self.gpus:
            lines.extend(self._generate_service(gpu))
        return "\n".join(lines)

    def _generate_header(self) -> list[str]:
        lines = ["# QWN (Qwen3.5) text deployment via vLLM"]
        if self._gpu_config_map:
            lines.append("# Per-GPU configs:")
            for gpu in self.gpus:
                config = self._get_gpu_config(gpu)
                lines.append(f"#   GPU {config.gpu_id}: {config.label}")
        else:
            quant_info = (
                f", quantization: {self.quantization}" if self.quantization else ""
            )
            lines.append(f"# Model: {self.model_name}{quant_info}")

        lines.extend(
            [
                f"# GPUs: {[gpu.index for gpu in self.gpus]}",
                "",
                f"name: {self.project_name}",
                "",
            ]
        )
        return lines

    def _generate_common_config(self) -> list[str]:
        lines = [
            "x-common-config: &common-config",
            "  restart: unless-stopped",
            "  volumes:",
            f"    - ${{HOME}}/{self.cache_hf}:/root/{self.cache_hf}",
            f"    - ${{HOME}}/{self.cache_vllm}:/root/{self.cache_vllm}",
            f"    - {self.data_dir}:/data",
        ]

        if self.mount_mode == "manual" and self.driver_lib_dir:
            lines.append(f"    - {self.driver_lib_dir}:/usr/local/nvidia/lib64:ro")

        lines.extend(
            [
                "  environment:",
                f"    - HF_ENDPOINT={self.network_config.hf_endpoint}",
                f"    - HF_HOME=/root/{self.cache_hf}",
                f"    - HF_HUB_CACHE=/root/{self.cache_hf_hub}",
                f"    - HUGGINGFACE_HUB_CACHE=/root/{self.cache_hf_hub}",
                f"    - PIP_INDEX_URL={self.network_config.pip_index_url}",
                f"    - PIP_TRUSTED_HOST={self.network_config.pip_trusted_host}",
                "    - VLLM_WORKER_MULTIPROC_METHOD=spawn",
            ]
        )

        if self.enable_sleep_mode:
            lines.append("    - VLLM_SERVER_DEV_MODE=1")

        if self._uses_runtime_offline_cache():
            lines.extend(
                [
                    "    - HF_HUB_OFFLINE=1",
                    "    - TRANSFORMERS_OFFLINE=1",
                ]
            )

        if self.network_config.runtime_http_proxy:
            lines.extend(
                [
                    f"    - HTTP_PROXY={self.network_config.runtime_http_proxy}",
                    f"    - http_proxy={self.network_config.runtime_http_proxy}",
                ]
            )
        if self.network_config.runtime_https_proxy:
            lines.extend(
                [
                    f"    - HTTPS_PROXY={self.network_config.runtime_https_proxy}",
                    f"    - https_proxy={self.network_config.runtime_https_proxy}",
                ]
            )
        if (
            self.network_config.runtime_http_proxy
            or self.network_config.runtime_https_proxy
        ):
            lines.extend(
                [
                    f"    - NO_PROXY={self.network_config.no_proxy_csv}",
                    f"    - no_proxy={self.network_config.no_proxy_csv}",
                ]
            )

        if self.mount_mode == "manual":
            lines.extend(
                [
                    "    - LD_LIBRARY_PATH=/usr/local/nvidia/lib64",
                    "    - CUDA_VISIBLE_DEVICES=0",
                    "    - NVIDIA_VISIBLE_DEVICES=0",
                ]
            )

        lines.extend(
            [
                "  ipc: host",
                "  ulimits:",
                "    memlock: -1",
                "    stack: 67108864",
                "",
            ]
        )
        return lines

    def _generate_service(self, gpu: GPUInfo) -> list[str]:
        gpu_config = self._get_gpu_config(gpu)
        service_port = self.port + gpu.index
        container_name = f"{self.project_name}--gpu{gpu.index}"
        serve_args = [
            "vllm",
            "serve",
            gpu_config.vllm_model_arg,
            "--served-model-name",
            gpu_config.served_model_name,
            "--reasoning-parser",
            "qwen3",
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--gpu-memory-utilization",
            str(gpu_config.gpu_memory_utilization),
            "--dtype",
            "half",
            "--trust-remote-code",
        ]

        if self.hf_token:
            serve_args.extend(["--api-key", self.hf_token])

        if gpu_config.quant_method == "bitsandbytes":
            serve_args.extend(
                [
                    "--quantization",
                    "bitsandbytes",
                    "--load-format",
                    "bitsandbytes",
                ]
            )

        if self.skip_mm_profiling:
            serve_args.append("--skip-mm-profiling")

        if self.enable_sleep_mode:
            serve_args.append("--enable-sleep-mode")

        if self.compilation_config:
            serve_args.extend(
                [
                    "--compilation-config",
                    json.dumps(self.compilation_config, separators=(",", ":")),
                ]
            )

        lines = [
            f"  qwn-gpu{gpu.index}:",
            "    <<: *common-config",
            f"    image: {gpu.image}",
            f"    container_name: {container_name}",
        ]

        lines.extend(
            [
                "    ports:",
                f'      - "{service_port}:{VLLM_INTERNAL_PORT}"',
                "    entrypoint:",
                "      - /bin/bash",
                "      - -lc",
                "    command:",
                f"      - exec {' '.join(shlex.quote(arg) for arg in serve_args)}",
            ]
        )

        if self.mount_mode == "manual":
            lines.append("    devices:")
            lines.append(f"      - /dev/nvidia{gpu.index}:/dev/nvidia{gpu.index}")
            for device in NvidiaDriverLibs.get_required_devices():
                if Path(device).exists():
                    lines.append(f"      - {device}:{device}")
        else:
            lines.extend(
                [
                    "    deploy:",
                    "      resources:",
                    "        reservations:",
                    "          devices:",
                    "            - driver: nvidia",
                    f'              device_ids: ["{gpu.index}"]',
                    "              capabilities: [gpu]",
                ]
            )

        lines.extend(
            [
                "    healthcheck:",
                f'      test: ["CMD-SHELL", "{HEALTHCHECK_TCP_PROBE}"]',
                f"      interval: {HEALTHCHECK_INTERVAL}",
                f"      timeout: {HEALTHCHECK_TIMEOUT}",
                f"      retries: {HEALTHCHECK_RETRIES}",
                f"      start_period: {HEALTHCHECK_START_PERIOD}",
                "",
            ]
        )
        return lines


class QWNComposer:
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
        hf_endpoint: str | None = None,
        pip_index_url: str | None = None,
        pip_trusted_host: str | None = None,
        quantization: Optional[str] = None,
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int = MAX_NUM_SEQS,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        skip_mm_profiling: bool = DEFAULT_SKIP_MM_PROFILING,
        enable_sleep_mode: bool = DEFAULT_ENABLE_SLEEP_MODE,
        cudagraph_mode: str | None = DEFAULT_CUDAGRAPH_MODE,
        cudagraph_capture_sizes: list[int] | None = None,
        gpu_layout: Optional[str] = None,
        gpu_configs: list[GpuModelConfig] | None = None,
    ):
        if gpu_layout and gpu_configs:
            raise ValueError("Use either gpu_layout or gpu_configs, not both")

        self.model_name = resolve_model_name(model_name)
        self.port = port
        self.gpu_ids = infer_gpu_ids(gpu_ids, gpu_configs)
        self.hf_token = hf_token
        self.mount_mode = mount_mode
        self.network_config = QWNNetworkConfig.from_overrides(
            proxy=http_proxy,
            hf_endpoint=hf_endpoint,
            pip_index_url=pip_index_url,
            pip_trusted_host=pip_trusted_host,
        )
        self.http_proxy = self.network_config.build_proxy
        self.quantization = normalize_model_key(quantization) if quantization else None
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.skip_mm_profiling = skip_mm_profiling
        self.enable_sleep_mode = enable_sleep_mode
        self.cudagraph_mode = cudagraph_mode.upper() if cudagraph_mode else None
        self.cudagraph_capture_sizes = cudagraph_capture_sizes
        self.gpu_layout = normalize_model_key(gpu_layout) if gpu_layout else None

        if self.gpu_layout:
            project_dash = re.sub(r"[^a-z0-9_-]", "-", self.gpu_layout)
            self.project_name = project_name or f"qwn-{project_dash}"
        elif gpu_configs:
            self.project_name = project_name or "qwn-multi"
        else:
            project_dash = self.model_name.replace("/", "--").lower()
            project_dash = re.sub(r"[^a-z0-9_-]", "_", project_dash)
            self.project_name = project_name or f"qwn--{project_dash}"

        if compose_dir:
            self.compose_dir = Path(compose_dir)
        else:
            self.compose_dir = Path(__file__).resolve().parent.parent / "configs"

        self.compose_file = self.compose_dir / f"{self.project_name}.yml"
        self.gpus = GPUDetector.detect(self.gpu_ids)
        self.gpu_configs = (
            build_gpu_configs_for_layout(self.gpu_layout, self.gpus)
            if self.gpu_layout
            else gpu_configs
        )
        for gpu in self.gpus:
            gpu.image = VLLM_IMAGE
        self.model_config_manager = ModelConfigManager()
        self.image_manager = DockerImageManager()

    def _find_compose_file(self) -> Path | None:
        if self.compose_file.exists():
            return self.compose_file

        multi = self.compose_dir / "qwn-multi.yml"
        if multi.exists():
            return multi

        candidates = sorted(self.compose_dir.glob("qwn*.yml"))
        if candidates:
            return candidates[0]
        return None

    @staticmethod
    def _find_qwn_containers(include_stopped: bool = False) -> list[str]:
        flag = "-a" if include_stopped else ""
        result = subprocess.run(
            f'docker ps {flag} --filter "name=qwn" --format "{{{{.Names}}}}"',
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        names = [
            name.strip()
            for name in result.stdout.splitlines()
            if name.strip() and re.match(r"qwn[-_]", name.strip())
        ]
        return sorted(names)

    def _resolve_compose_cmd(self, cmd: str) -> bool:
        compose_file = self._find_compose_file()
        if not compose_file:
            return False
        if compose_file != self.compose_file:
            logger.mesg(f"[qwn] Using compose file: {compose_file}")
        full_cmd = f"docker compose -f {compose_file} {cmd}"
        logger.mesg(f"[qwn] Running: {full_cmd}")
        subprocess.run(full_cmd, shell=True)
        return True

    def _get_container_name(self, gpu: GPUInfo) -> str:
        return f"{self.project_name}--gpu{gpu.index}"

    def _ensure_compose_dir(self) -> None:
        self.compose_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_data_dir(self) -> Path:
        data_dir = Path(__file__).resolve().parent / "docker_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def generate_compose_file(self) -> Path:
        self._ensure_compose_dir()
        data_dir = self._ensure_data_dir()
        generator = ComposeFileGenerator(
            gpus=self.gpus,
            model_name=self.model_name,
            port=self.port,
            project_name=self.project_name,
            data_dir=data_dir,
            hf_token=self.hf_token,
            hf_endpoint=self.network_config.hf_endpoint,
            mount_mode=self.mount_mode,
            http_proxy=self.network_config.build_proxy,
            pip_index_url=self.network_config.pip_index_url,
            pip_trusted_host=self.network_config.pip_trusted_host,
            quantization=self.quantization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            skip_mm_profiling=self.skip_mm_profiling,
            enable_sleep_mode=self.enable_sleep_mode,
            cudagraph_mode=self.cudagraph_mode,
            cudagraph_capture_sizes=self.cudagraph_capture_sizes,
            gpu_configs=self.gpu_configs,
            network_config=self.network_config,
        )
        self.compose_file.write_text(generator.generate())
        logger.okay(f"[qwn] Generated: {self.compose_file}")
        return self.compose_file

    def get_backend_endpoints(self) -> list[str]:
        return [f"http://localhost:{self.port + gpu.index}" for gpu in self.gpus]

    def get_backend_container_names(self) -> list[str]:
        return [f"{self.project_name}--gpu{gpu.index}" for gpu in self.gpus]

    @staticmethod
    def _extract_host_port(ports_str: str) -> Optional[int]:
        match = re.search(r"(?:0\.0\.0\.0|::):(\d+)->", ports_str)
        if match:
            return int(match.group(1))
        return None

    def _discover_running_backend_endpoints(self) -> list[str]:
        result = subprocess.run(
            "docker ps --format '{{.Names}}|{{.Ports}}'",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        prefix = f"{self.project_name}--gpu"
        endpoints: list[tuple[str, int]] = []
        for line in result.stdout.splitlines():
            if not line.strip() or "|" not in line:
                continue
            container_name, ports = line.split("|", 1)
            container_name = container_name.strip()
            if not container_name.startswith(prefix):
                continue
            host_port = self._extract_host_port(ports)
            if host_port is None:
                continue
            endpoints.append((container_name, host_port))

        endpoints.sort()
        return [f"http://localhost:{port}" for _, port in endpoints]

    def _get_control_endpoints(self) -> list[str]:
        return (
            self._discover_running_backend_endpoints() or self.get_backend_endpoints()
        )

    def _request_backend_control(
        self,
        path: str,
        *,
        method: str = "POST",
        params: dict[str, object] | None = None,
    ) -> tuple[list[tuple[str, dict | None]], list[str], list[tuple[str, str]]]:
        endpoints = self._get_control_endpoints()
        if not endpoints:
            return [], [], []

        successes: list[tuple[str, dict | None]] = []
        unsupported: list[str] = []
        failures: list[tuple[str, str]] = []
        for endpoint in endpoints:
            url = f"{endpoint}{path}"
            try:
                response = requests.request(
                    method,
                    url,
                    params=params,
                    timeout=SLEEP_CONTROL_TIMEOUT_SEC,
                )
            except requests.RequestException as exc:
                failures.append((endpoint, str(exc)))
                continue

            if response.status_code == 200:
                payload = None
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type.lower():
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = None
                successes.append((endpoint, payload))
            elif response.status_code == 404:
                unsupported.append(endpoint)
            else:
                failures.append((endpoint, f"HTTP {response.status_code}"))

        return successes, unsupported, failures

    def sleep(
        self,
        level: int = DEFAULT_SLEEP_LEVEL,
        mode: str = DEFAULT_SLEEP_MODE,
    ) -> bool:
        successes, unsupported, failures = self._request_backend_control(
            "/sleep",
            params={"level": level, "mode": mode},
        )

        if successes:
            set_backend_sleep_states(
                [endpoint for endpoint, _ in successes], sleeping=True
            )
            logger.okay(
                f"[qwn] Requested sleep(level={level}, mode={mode}) on {len(successes)} backend(s)"
            )
        elif unsupported:
            logger.warn(
                "[qwn] Sleep endpoints are unavailable. Redeploy with --enable-sleep-mode to use fast wake/resume."
            )
        else:
            logger.warn("[qwn] Failed to contact any running backend for sleep")

        for endpoint, error in failures:
            logger.warn(f"[qwn] Sleep request failed for {endpoint}: {error}")
        return bool(successes)

    def wake(self, wait_healthy: bool = False) -> bool:
        successes, unsupported, failures = self._request_backend_control("/wake_up")

        if successes:
            set_backend_sleep_states(
                [endpoint for endpoint, _ in successes],
                sleeping=False,
            )
            logger.okay(f"[qwn] Requested wake-up on {len(successes)} backend(s)")
        elif unsupported:
            logger.warn(
                "[qwn] Wake-up endpoints are unavailable. Redeploy with --enable-sleep-mode to use fast wake/resume."
            )
        else:
            logger.warn("[qwn] Failed to contact any running backend for wake-up")

        for endpoint, error in failures:
            logger.warn(f"[qwn] Wake request failed for {endpoint}: {error}")

        if wait_healthy and successes:
            healthy = self.wait_for_healthy_backends(
                timeout_sec=SLEEP_WAKE_TIMEOUT_SEC,
                poll_interval_sec=SLEEP_WAKE_POLL_INTERVAL_SEC,
                label="[qwn]",
            )
            if healthy:
                logger.okay("[qwn] Backends are healthy after wake-up")
            else:
                logger.warn(
                    "[qwn] Wake-up requested, but healthy backends did not appear in time"
                )
            return healthy

        return bool(successes)

    def sleep_status(self) -> bool:
        successes, unsupported, failures = self._request_backend_control(
            "/is_sleeping",
            method="GET",
        )

        if not successes and unsupported:
            logger.warn(
                "[qwn] Sleep status endpoints are unavailable. Redeploy with --enable-sleep-mode to inspect sleep state."
            )
            return False

        if not successes and not failures:
            logger.warn("[qwn] No running backends found for sleep status")
            return False

        for endpoint, payload in successes:
            sleeping = bool((payload or {}).get("is_sleeping", False))
            update_backend_sleep_states({endpoint: sleeping})
            state = "sleeping" if sleeping else "awake"
            logger.mesg(f"[qwn] {endpoint}: {state}")

        for endpoint, error in failures:
            logger.warn(f"[qwn] Sleep status failed for {endpoint}: {error}")

        return bool(successes)

    @staticmethod
    def _build_warmup_payload(model_id: str) -> dict[str, object]:
        return {
            "model": model_id,
            "messages": [{"role": "user", "content": WARMUP_PROMPT}],
            "max_tokens": WARMUP_MAX_TOKENS,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    @staticmethod
    def _discover_backend_model_id(endpoint: str) -> str:
        response = requests.get(
            f"{endpoint}/v1/models",
            timeout=SLEEP_CONTROL_TIMEOUT_SEC,
        )
        if response.status_code != 200:
            raise RuntimeError(f"model discovery returned HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("model discovery returned invalid JSON") from exc

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("model discovery returned no models")

        model_id = str(data[0].get("id", "")).strip()
        if not model_id:
            raise RuntimeError("model discovery returned an empty model id")
        return model_id

    def _warmup_backend(self, endpoint: str) -> str:
        model_id = self._discover_backend_model_id(endpoint)
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=self._build_warmup_payload(model_id),
            timeout=WARMUP_REQUEST_TIMEOUT_SEC,
        )
        if response.status_code != 200:
            raise RuntimeError(f"warmup returned HTTP {response.status_code}")
        return model_id

    def warmup(self, wait_healthy: bool = False) -> bool:
        if wait_healthy:
            healthy = self.wait_for_healthy_backends(
                timeout_sec=WARMUP_HEALTH_TIMEOUT_SEC,
                poll_interval_sec=WARMUP_HEALTH_POLL_INTERVAL_SEC,
                label="[qwn]",
            )
            if not healthy:
                return False

        endpoints = self._get_control_endpoints()
        if not endpoints:
            logger.warn("[qwn] No running backends found for warmup")
            return False

        successes: list[tuple[str, str]] = []
        failures: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=min(len(endpoints), 8)) as executor:
            future_map = {
                executor.submit(self._warmup_backend, endpoint): endpoint
                for endpoint in endpoints
            }
            for future in as_completed(future_map):
                endpoint = future_map[future]
                try:
                    model_id = future.result()
                except Exception as exc:
                    failures.append((endpoint, str(exc)))
                    continue
                successes.append((endpoint, model_id))

        for endpoint, model_id in successes:
            logger.mesg(f"[qwn] Warmed {endpoint} with model {model_id}")
        for endpoint, error in failures:
            logger.warn(f"[qwn] Warmup failed for {endpoint}: {error}")

        if successes and not failures:
            logger.okay(f"[qwn] Warmed up {len(successes)} backend(s)")
        elif successes:
            logger.warn(
                f"[qwn] Warmed {len(successes)} backend(s), but {len(failures)} backend(s) failed"
            )
        else:
            logger.warn("[qwn] Failed to warm any running backend")

        return bool(successes) and not failures

    def wait_for_healthy_backends(
        self,
        timeout_sec: float = 300.0,
        poll_interval_sec: float = 5.0,
        label: str = "[qwn]",
    ) -> bool:
        healthy = wait_for_healthy_docker_containers(
            self.get_backend_container_names(),
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            label=label,
        )
        if healthy:
            set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)
        return healthy

    def _run_compose_cmd(
        self,
        cmd: str,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        full_cmd = f"docker compose -f {self.compose_file} {cmd}"
        logger.mesg(f"[qwn] Running: {full_cmd}")
        return subprocess.run(full_cmd, shell=True, capture_output=capture_output)

    def up(self) -> None:
        if not self.gpus:
            logger.warn("× No healthy GPUs detected")
            summary = GPUDetector.get_unhealthy_gpu_summary()
            if summary:
                logger.warn(summary)
            return

        logger.mesg(f"[qwn] Starting vLLM for model: {self.model_name}")
        if self.gpu_configs:
            for config in self.gpu_configs:
                logger.mesg(f"[qwn]   GPU {config.gpu_id}: {config.label}")

        if not self.image_manager.ensure_image(
            VLLM_IMAGE,
            network_config=self.network_config,
        ):
            return

        self._ensure_data_dir()
        self.generate_compose_file()
        result = self._run_compose_cmd("up -d --remove-orphans", capture_output=True)
        if result.returncode != 0:
            stderr = (
                result.stderr.decode("utf-8", errors="replace")
                if isinstance(result.stderr, bytes)
                else result.stderr
            )
            logger.warn(f"[qwn] Startup failed: {stderr.strip()}")
            return
        logger.okay(
            f"[qwn] Started services on GPUs: {[gpu.index for gpu in self.gpus]}"
        )
        set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)
        self.ps()

    def down(self) -> None:
        if self._resolve_compose_cmd("down --remove-orphans"):
            return

        containers = self._find_qwn_containers(include_stopped=True)
        if not containers:
            logger.mesg("[qwn] No QWN containers found")
            return
        logger.mesg(f"[qwn] Removing {len(containers)} container(s)")
        for container in containers:
            subprocess.run(f"docker rm -f {container}", shell=True, capture_output=True)
        logger.okay(f"[qwn] Removed {len(containers)} container(s)")
        clear_backend_sleep_states(self.get_backend_endpoints())

    def stop(self) -> None:
        if self._resolve_compose_cmd("stop"):
            return

        containers = self._find_qwn_containers(include_stopped=False)
        if not containers:
            logger.mesg("[qwn] No running QWN containers found")
            return
        for container in containers:
            subprocess.run(f"docker stop {container}", shell=True, capture_output=True)
        logger.okay(f"[qwn] Stopped {len(containers)} container(s)")

    def start(self) -> None:
        if self._resolve_compose_cmd("start"):
            return

        containers = self._find_qwn_containers(include_stopped=True)
        running = set(self._find_qwn_containers(include_stopped=False))
        stopped = [container for container in containers if container not in running]
        if not stopped:
            logger.mesg("[qwn] No stopped QWN containers found")
            return
        for container in stopped:
            subprocess.run(f"docker start {container}", shell=True, capture_output=True)
        logger.okay(f"[qwn] Started {len(stopped)} container(s)")
        set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)

    def restart(self) -> None:
        if self._resolve_compose_cmd("restart"):
            return

        containers = self._find_qwn_containers(include_stopped=False)
        if not containers:
            logger.mesg("[qwn] No running QWN containers found")
            return
        for container in containers:
            subprocess.run(
                f"docker restart {container}", shell=True, capture_output=True
            )
        logger.okay(f"[qwn] Restarted {len(containers)} container(s)")
        set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)

    def ps(self) -> None:
        if self._resolve_compose_cmd("ps"):
            return
        self._show_manual_status()

    def logs(self, follow: bool = False, tail: int = 100) -> None:
        follow_flag = "-f" if follow else ""
        cmd = f"logs --tail={tail} {follow_flag}".strip()
        if self._resolve_compose_cmd(cmd):
            return

        containers = self._find_qwn_containers(include_stopped=True)
        if not containers:
            logger.mesg("[qwn] No QWN containers found")
            return
        for container in containers:
            logger.mesg(f"[qwn] Logs for {container}:")
            subprocess.run(
                f"docker logs --tail={tail} {follow_flag} {container}".strip(),
                shell=True,
            )

    def _show_manual_status(self) -> None:
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

            healthy, _ = GPUDetector.check_gpu_health(gpu.index)
            gpu_health = "healthy" if healthy else "unhealthy"
            print(
                f"{gpu.index:<5} {container_name:<42} {container_port:<7} {container_status:<12} {gpu_health:<10}"
            )

        print("=" * 85)

    def health(self) -> None:
        print("=" * 70)
        print(f"{'GPU':<6} {'STATUS':<12} {'INFO':<50}")
        print("-" * 70)
        for gpu in self.gpus:
            healthy, message = GPUDetector.check_gpu_health(gpu.index)
            status = "healthy" if healthy else "unhealthy"
            info = message[:48] + ".." if len(message) > 50 else message
            print(f"{gpu.index:<6} {status:<12} {info:<50}")
        print("=" * 70)


CLI_EPILOG = """
Examples:
  qwn compose up
  qwn compose up -g 0,1
    qwn compose up --gpu-layout uniform-awq
    qwn compose up --gpu-configs "0,1"
  qwn compose up --gpu-configs "0:4b:4bit,1:4b:4bit"
  qwn compose generate -j qwn-awq --gpu-configs "0:4b:4bit"
        qwn compose warmup --wait-healthy
    qwn compose sleep
    qwn compose wake --wait-healthy
    qwn compose sleep-status
  qwn compose logs -f
  qwn compose down
"""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Manage QWN vLLM Docker compose deployments"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = CLI_EPILOG

    subparsers = parser.add_subparsers(dest="compose_action", required=True)

    def add_common_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "-m",
            "--model-name",
            type=str,
            default=MODEL_NAME,
            help=f"Model name (default: {MODEL_NAME})",
        )
        target.add_argument(
            "-p",
            "--port",
            type=int,
            default=SERVER_PORT,
            help=f"Starting port (default: {SERVER_PORT})",
        )
        target.add_argument(
            "-j",
            "--project-name",
            type=str,
            default=None,
            help="Compose project name",
        )
        target.add_argument(
            "-g",
            "--gpus",
            type=str,
            default=None,
            help="Comma-separated GPU IDs (default: all healthy GPUs)",
        )

    def add_targeting_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--gpu-configs",
            type=str,
            default=None,
            help='Per-GPU config: "GPU[:MODEL[:QUANT]],..."',
        )
        target.add_argument(
            "--gpu-layout",
            choices=[GPU_LAYOUT_UNIFORM_AWQ],
            default=None,
            help="Named multi-GPU layout preset",
        )

    def add_deployment_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("-t", "--hf-token", type=str, default=None)
        target.add_argument(
            "--mount-mode",
            choices=["nvidia-runtime", "manual"],
            default=DEVICE_MOUNT_MODE,
        )
        target.add_argument("--proxy", type=str, default=None)
        target.add_argument("--hf-endpoint", type=str, default=None)
        target.add_argument("--pip-index-url", type=str, default=None)
        target.add_argument("--pip-trusted-host", type=str, default=None)
        target.add_argument(
            "-q",
            "--quantization",
            choices=["none", "awq", "bitsandbytes"],
            default=None,
        )
        target.add_argument(
            "--max-model-len",
            type=int,
            default=MAX_MODEL_LEN,
        )
        target.add_argument(
            "--max-num-seqs",
            type=int,
            default=MAX_NUM_SEQS,
        )
        target.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=GPU_MEMORY_UTILIZATION,
        )
        target.set_defaults(skip_mm_profiling=DEFAULT_SKIP_MM_PROFILING)
        target.add_argument(
            "--skip-mm-profiling",
            dest="skip_mm_profiling",
            action="store_true",
            help="Skip multimodal memory profiling during startup to reduce readiness time",
        )
        target.add_argument(
            "--no-skip-mm-profiling",
            dest="skip_mm_profiling",
            action="store_false",
            help="Keep vLLM multimodal memory profiling enabled during startup",
        )
        target.set_defaults(enable_sleep_mode=DEFAULT_ENABLE_SLEEP_MODE)
        target.add_argument(
            "--enable-sleep-mode",
            dest="enable_sleep_mode",
            action="store_true",
            help=(
                "Enable vLLM sleep mode endpoints so running backends can be "
                "put to sleep and woken up quickly without a full cold restart"
            ),
        )
        target.add_argument(
            "--no-enable-sleep-mode",
            dest="enable_sleep_mode",
            action="store_false",
            help="Disable vLLM sleep mode endpoints",
        )
        target.add_argument(
            "--cudagraph-mode",
            type=str.upper,
            choices=CUDAGRAPH_MODE_CHOICES,
            default=DEFAULT_CUDAGRAPH_MODE,
            help=(
                "Experimental vLLM cudagraph mode override. If unset, keep the "
                "runtime's default compilation behavior."
            ),
        )
        target.add_argument(
            "--cudagraph-capture-sizes",
            type=parse_cudagraph_capture_sizes,
            default=None,
            help=(
                "Comma-separated cudagraph batch sizes. Only used together with "
                "--cudagraph-mode; otherwise ignored."
            ),
        )
        add_targeting_arguments(target)

    parser_up = subparsers.add_parser("up", help="Start QWN containers")
    add_common_arguments(parser_up)
    add_deployment_arguments(parser_up)

    parser_generate = subparsers.add_parser("generate", help="Generate compose file")
    add_common_arguments(parser_generate)
    add_deployment_arguments(parser_generate)

    for action in ("down", "stop", "start", "restart", "ps", "health"):
        action_parser = subparsers.add_parser(
            action, help=f"{action.title()} QWN containers"
        )
        add_common_arguments(action_parser)

    parser_sleep = subparsers.add_parser(
        "sleep",
        help="Put running QWN backends into vLLM sleep mode",
    )
    add_common_arguments(parser_sleep)
    add_targeting_arguments(parser_sleep)
    parser_sleep.add_argument(
        "--sleep-level",
        type=int,
        choices=[0, 1, 2],
        default=DEFAULT_SLEEP_LEVEL,
        help=(
            "vLLM sleep level: 0=pause scheduling, 1=offload weights, "
            "2=drop all GPU memory"
        ),
    )
    parser_sleep.add_argument(
        "--sleep-mode",
        choices=["abort", "wait"],
        default=DEFAULT_SLEEP_MODE,
        help="How to handle in-flight requests when entering sleep mode",
    )

    parser_wake = subparsers.add_parser(
        "wake",
        help="Wake running QWN backends from vLLM sleep mode",
    )
    add_common_arguments(parser_wake)
    add_targeting_arguments(parser_wake)
    parser_wake.add_argument(
        "--wait-healthy",
        action="store_true",
        help="Wait for backend /health endpoints to return healthy after wake-up",
    )

    parser_sleep_status = subparsers.add_parser(
        "sleep-status",
        help="Show vLLM sleep state for running QWN backends",
    )
    add_common_arguments(parser_sleep_status)
    add_targeting_arguments(parser_sleep_status)

    parser_warmup = subparsers.add_parser(
        "warmup",
        help="Warm QWN backends with a tiny generation request to remove first-request TTFT spikes",
    )
    add_common_arguments(parser_warmup)
    add_targeting_arguments(parser_warmup)
    parser_warmup.add_argument(
        "--wait-healthy",
        action="store_true",
        help="Wait for backend /health endpoints before sending warmup requests",
    )

    parser_logs = subparsers.add_parser("logs", help="View container logs")
    add_common_arguments(parser_logs)
    parser_logs.add_argument("-f", "--follow", action="store_true")
    parser_logs.add_argument("--tail", type=int, default=100)


def _composer_from_args(args: argparse.Namespace) -> QWNComposer:
    composer_kwargs = {
        "model_name": getattr(args, "model_name", MODEL_NAME),
        "port": getattr(args, "port", SERVER_PORT),
        "project_name": getattr(args, "project_name", None),
        "gpu_ids": getattr(args, "gpus", None),
        "hf_token": getattr(args, "hf_token", None),
        "mount_mode": getattr(args, "mount_mode", DEVICE_MOUNT_MODE),
        "http_proxy": getattr(args, "proxy", None),
        "hf_endpoint": getattr(args, "hf_endpoint", None),
        "pip_index_url": getattr(args, "pip_index_url", None),
        "pip_trusted_host": getattr(args, "pip_trusted_host", None),
        "quantization": getattr(args, "quantization", None),
        "max_model_len": getattr(args, "max_model_len", MAX_MODEL_LEN),
        "max_num_seqs": getattr(args, "max_num_seqs", MAX_NUM_SEQS),
        "gpu_memory_utilization": getattr(
            args,
            "gpu_memory_utilization",
            GPU_MEMORY_UTILIZATION,
        ),
        "skip_mm_profiling": getattr(
            args,
            "skip_mm_profiling",
            DEFAULT_SKIP_MM_PROFILING,
        ),
        "enable_sleep_mode": getattr(
            args,
            "enable_sleep_mode",
            DEFAULT_ENABLE_SLEEP_MODE,
        ),
        "cudagraph_mode": getattr(
            args,
            "cudagraph_mode",
            DEFAULT_CUDAGRAPH_MODE,
        ),
        "cudagraph_capture_sizes": getattr(
            args,
            "cudagraph_capture_sizes",
            None,
        ),
    }
    gpu_configs = getattr(args, "gpu_configs", None)
    if gpu_configs:
        composer_kwargs["gpu_configs"] = parse_gpu_configs(gpu_configs)
    composer_kwargs["gpu_layout"] = getattr(args, "gpu_layout", None)
    return QWNComposer(**composer_kwargs)


def run_from_args(args: argparse.Namespace) -> None:
    composer = _composer_from_args(args)
    action = args.compose_action
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
    elif action == "sleep":
        composer.sleep(
            level=getattr(args, "sleep_level", DEFAULT_SLEEP_LEVEL),
            mode=getattr(args, "sleep_mode", DEFAULT_SLEEP_MODE),
        )
    elif action == "wake":
        composer.wake(wait_healthy=getattr(args, "wait_healthy", False))
    elif action == "sleep-status":
        composer.sleep_status()
    elif action == "warmup":
        composer.warmup(wait_healthy=getattr(args, "wait_healthy", False))
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
    else:
        raise ValueError(f"Unknown compose action: {action}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
