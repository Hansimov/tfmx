"""QSR Docker Compose manager.

Deploys Qwen3-ASR models with vLLM, one container per GPU, and provides
Docker lifecycle helpers used by the unified ``qsr`` CLI.
"""

import argparse
import io
import json
import math
import mimetypes
import re
import shlex
import subprocess
import time
import wave

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from tclogger import logger

from ..utils.service_bootstrap import docker_status_to_health
from ..utils.service_bootstrap import get_docker_container_statuses
from ..utils.service_bootstrap import wait_for_healthy_http_endpoints
from ..utils.service_bootstrap import wait_for_healthy_docker_containers
from .networking import DEFAULT_HF_ENDPOINT
from .networking import DEFAULT_PIP_INDEX_URL
from .networking import DEFAULT_PIP_TRUSTED_HOST
from .networking import QSRNetworkConfig


SERVER_PORT = 27980
MACHINE_PORT = 27900

MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"

HF_ENDPOINT = DEFAULT_HF_ENDPOINT
CACHE_HF = ".cache/huggingface"
CACHE_HF_HUB = f"{CACHE_HF}/hub"
CACHE_VLLM = ".cache/vllm"

QWEN_ASR_VLLM_VERSION = "0.19.0"
VLLM_BASE_IMAGE = f"vllm/vllm-openai:v{QWEN_ASR_VLLM_VERSION}"
VLLM_IMAGE = "tfmx-vllm-openai:qwen3-asr-v0.19.0"
VLLM_IMAGE_MIRROR = "m.daocloud.io"
VLLM_INTERNAL_PORT = 8000
VLLM_AUDIO_SPEC = f"vllm[audio]=={QWEN_ASR_VLLM_VERSION}"
QWEN_ASR_SPEC = "qwen-asr==0.0.6"
QWEN_ASR_RUNTIME_DEPS = [
    "accelerate==1.12.0",
    "qwen-omni-utils",
    "librosa",
    "soundfile",
    "sox",
    "nagisa==0.2.11",
    "soynlp==0.0.493",
]
QWEN_ASR_RUNTIME_DEPS_ARG = " ".join(QWEN_ASR_RUNTIME_DEPS)
QWEN_ASR_HF_HUB_SPEC = "huggingface-hub>=0.34.0,<1.0"
QWEN_ASR_DOCKERFILE = "Dockerfile.qwen3-asr-vllm"

MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8
GPU_MEMORY_UTILIZATION = 0.35
DEVICE_MOUNT_MODE = "manual"
GPU_LAYOUT_UNIFORM = "uniform"
DEFAULT_SKIP_MM_PROFILING = False
DEFAULT_CUDAGRAPH_MODE: str | None = None
DEFAULT_ENABLE_SLEEP_MODE = False
DEFAULT_SLEEP_LEVEL = 1
DEFAULT_SLEEP_MODE = "abort"
SLEEP_CONTROL_TIMEOUT_SEC = 5.0
SLEEP_WAKE_TIMEOUT_SEC = 180.0
SLEEP_WAKE_POLL_INTERVAL_SEC = 2.0

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
HEALTHCHECK_START_PERIOD = "180s"
HEALTHCHECK_TCP_PROBE = f"bash -lc 'exec 3<>/dev/tcp/127.0.0.1/{VLLM_INTERNAL_PORT}'"
WARMUP_WAIT_TIMEOUT_SEC = 300.0
WARMUP_POLL_INTERVAL_SEC = 1.0
READINESS_REQUEST_TIMEOUT_SEC = 1.0
WARMUP_REQUEST_TIMEOUT_SEC = 120.0
SLEEP_STATE_FILE = Path.home() / ".cache" / "tfmx" / "qsr_sleep_state.json"

SUPPORTED_MODELS = {
    MODEL_NAME: {
        "size": "0.6B",
        "family": "qwen3-asr",
        "type": "asr",
        "max_model_len": MAX_MODEL_LEN,
    }
}

MODEL_SHORTCUTS = {
    "0.6b": MODEL_NAME,
    "qwen3-asr": MODEL_NAME,
    "qwen3-asr-0.6b": MODEL_NAME,
    "default": MODEL_NAME,
}

MODEL_SHORTCUT_REV = {MODEL_NAME: "0.6b"}
MODEL_SHORTCUT_REV_LOWER = {MODEL_NAME.lower(): "0.6b"}

DISPLAY_SHORTCUTS = {
    "0.6b": "0.6B",
}

GPU_COMPUTE_CAPS = {
    "8.6": "RTX 30xx",
    "8.9": "RTX 40xx",
    "8.0": "A100/A30",
    "9.0": "H100",
}


def normalize_model_key(key: str) -> str:
    return key.strip().lower() if key else ""


def resolve_model_name(key: str) -> str:
    if not key:
        return MODEL_NAME

    normalized = normalize_model_key(key)
    if normalized in MODEL_SHORTCUTS:
        return MODEL_SHORTCUTS[normalized]

    for full_name in SUPPORTED_MODELS:
        if full_name.lower() == normalized:
            return full_name

    return key


def get_model_shortcut(model_name: str) -> str:
    if not model_name:
        return ""

    direct = MODEL_SHORTCUT_REV.get(model_name)
    if direct:
        return direct

    normalized = normalize_model_key(model_name)
    lower_direct = MODEL_SHORTCUT_REV_LOWER.get(normalized)
    if lower_direct:
        return lower_direct
    if normalized in MODEL_SHORTCUTS:
        resolved_model = MODEL_SHORTCUTS[normalized]
        return MODEL_SHORTCUT_REV.get(resolved_model, normalized)

    tail = normalized.split("/")[-1]
    match = re.search(r"(\d+(?:\.\d+)?b)", tail)
    if match:
        return match.group(1)
    return tail or model_name


def get_display_shortcut(shortcut: str) -> str:
    return DISPLAY_SHORTCUTS.get(normalize_model_key(shortcut), shortcut)


def get_model_api_aliases(model_name: str) -> list[str]:
    shortcut = normalize_model_key(get_model_shortcut(model_name))
    aliases: list[str] = []
    if shortcut == "0.6b":
        aliases.extend(["qwen3-asr-0.6b", "qwen3-asr"])

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
class StartupProfile:
    image_ready_elapsed_sec: float = 0.0
    compose_generate_elapsed_sec: float = 0.0
    compose_up_elapsed_sec: float = 0.0
    container_health_elapsed_sec: float | None = None
    endpoint_ready_elapsed_sec: float | None = None
    warmup_elapsed_sec: float | None = None
    total_elapsed_sec: float = 0.0
    endpoint_by_container: dict[str, str] = field(default_factory=dict)
    container_health_by_container: dict[str, float] = field(default_factory=dict)
    endpoint_ready_by_container: dict[str, float] = field(default_factory=dict)
    warmup_by_container: dict[str, float] = field(default_factory=dict)
    warmup_result_by_container: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _format_seconds(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.2f}s"

    def log(self, label: str = "[qsr]") -> None:
        logger.note(f"{label} Startup profile:")
        logger.mesg(
            f"{label}   image-ready={self.image_ready_elapsed_sec:.2f}s | "
            f"compose-generate={self.compose_generate_elapsed_sec:.2f}s | "
            f"compose-up={self.compose_up_elapsed_sec:.2f}s | "
            f"docker-healthy={self._format_seconds(self.container_health_elapsed_sec)} | "
            f"http-ready={self._format_seconds(self.endpoint_ready_elapsed_sec)} | "
            f"warmup={self._format_seconds(self.warmup_elapsed_sec)} | "
            f"total={self.total_elapsed_sec:.2f}s"
        )

        container_names = sorted(
            {
                *self.endpoint_by_container.keys(),
                *self.container_health_by_container.keys(),
                *self.endpoint_ready_by_container.keys(),
                *self.warmup_by_container.keys(),
                *self.warmup_result_by_container.keys(),
            }
        )
        for container_name in container_names:
            endpoint = self.endpoint_by_container.get(container_name, "")
            details = []
            if endpoint:
                details.append(endpoint)
            if container_name in self.container_health_by_container:
                details.append(
                    f"docker={self.container_health_by_container[container_name]:.2f}s"
                )
            if container_name in self.endpoint_ready_by_container:
                details.append(
                    f"http={self.endpoint_ready_by_container[container_name]:.2f}s"
                )
            if container_name in self.warmup_by_container:
                details.append(
                    f"warmup={self.warmup_by_container[container_name]:.2f}s"
                )
            if container_name in self.warmup_result_by_container:
                details.append(self.warmup_result_by_container[container_name])
            logger.mesg(f"{label}   {container_name}: {' | '.join(details)}")


@dataclass
class GpuModelConfig:
    gpu_id: int
    model_name: str = MODEL_NAME
    served_model_name: str = ""

    def __post_init__(self) -> None:
        self.model_name = resolve_model_name(self.model_name)
        if not self.served_model_name:
            self.served_model_name = (
                get_model_shortcut(self.model_name) or self.model_name
            )

    @property
    def model_shortcut(self) -> str:
        return get_model_shortcut(self.model_name)

    @property
    def display_shortcut(self) -> str:
        return get_display_shortcut(self.model_shortcut)

    @property
    def label(self) -> str:
        return self.served_model_name or self.model_shortcut or self.model_name

    @property
    def display_label(self) -> str:
        if self.served_model_name and self.served_model_name != self.model_shortcut:
            return self.served_model_name
        return self.display_shortcut or self.label

    def to_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "model_name": self.model_name,
            "model_shortcut": self.model_shortcut,
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
            raise ValueError(f"Invalid config: '{part}'. Format: GPU_ID[:MODEL]")
        if len(fields) > 2:
            raise ValueError(f"Invalid config: '{part}'. Format: GPU_ID[:MODEL]")

        gpu_id = int(fields[0])
        if gpu_id in seen_gpu_ids:
            raise ValueError(f"Duplicate GPU ID in config: '{gpu_id}'")
        seen_gpu_ids.add(gpu_id)
        model_name = (
            resolve_model_name(fields[1])
            if len(fields) > 1 and fields[1]
            else MODEL_NAME
        )
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


def build_gpu_configs_for_layout(
    layout: str,
    gpus: list["GPUInfo"],
) -> list[GpuModelConfig]:
    normalized_layout = normalize_model_key(layout)
    if normalized_layout != GPU_LAYOUT_UNIFORM:
        raise ValueError(f"Unsupported GPU layout: {layout}")

    return [GpuModelConfig(gpu_id=gpu.index, model_name=MODEL_NAME) for gpu in gpus]


def build_compilation_config(
    cudagraph_mode: str | None = DEFAULT_CUDAGRAPH_MODE,
) -> dict[str, object] | None:
    if not cudagraph_mode:
        return None

    mode = cudagraph_mode.upper()
    if mode not in CUDAGRAPH_MODE_CHOICES:
        raise ValueError(f"Unsupported cudagraph mode: {cudagraph_mode}")
    return {"cudagraph_mode": mode}


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
                        f"[qsr] GPU {gpu.index} excluded (unhealthy): {message}"
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
        return {
            "size": "0.6B",
            "family": "unknown",
            "type": "asr",
            "max_model_len": MAX_MODEL_LEN,
        }


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

        logger.mesg(f"[qsr] Pulling image: {image}")
        try:
            subprocess.run(f"docker pull {image}", shell=True, check=True)
            return True
        except subprocess.CalledProcessError:
            pass

        mirror_image = f"{VLLM_IMAGE_MIRROR}/{image}"
        logger.mesg(f"[qsr] Pulling image from mirror: {mirror_image}")
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
    def ensure_qwen_asr_image(
        image: str = VLLM_IMAGE,
        network_config: QSRNetworkConfig | None = None,
    ) -> bool:
        network_config = network_config or QSRNetworkConfig.from_overrides()
        if DockerImageManager._image_exists(image):
            return True

        if not DockerImageManager.ensure_base_image(VLLM_BASE_IMAGE):
            return False

        dockerfile_path = DockerImageManager._docker_data_dir() / QWEN_ASR_DOCKERFILE
        if not dockerfile_path.exists():
            logger.warn(f"× Missing runtime Dockerfile: {dockerfile_path}")
            return False

        logger.mesg(
            f"[qsr] Building runtime image: {image} (base={VLLM_BASE_IMAGE}, vllm_audio={VLLM_AUDIO_SPEC}, qwen_asr={QWEN_ASR_SPEC})"
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
            f"VLLM_AUDIO_SPEC={VLLM_AUDIO_SPEC}",
            "--build-arg",
            f"QWEN_ASR_SPEC={QWEN_ASR_SPEC}",
            "--build-arg",
            f"QWEN_ASR_RUNTIME_DEPS={QWEN_ASR_RUNTIME_DEPS_ARG}",
            "--build-arg",
            f"HF_HUB_SPEC={QWEN_ASR_HF_HUB_SPEC}",
        ]
        build_cmd.extend(network_config.docker_build_args())

        if network_config.use_host_network_for_build:
            logger.mesg("[qsr] Docker build uses host network for loopback proxy")
            build_cmd.extend(["--network", "host"])

        build_cmd.extend(["-f", str(dockerfile_path), str(dockerfile_path.parent)])
        result = subprocess.run(build_cmd, text=True)
        if result.returncode != 0:
            logger.warn(
                f"× Failed to build runtime image: docker build exited with {result.returncode}"
            )
            return False

        logger.okay(f"[qsr] Built runtime image: {image}")
        return True

    @staticmethod
    def ensure_image(
        image: str = VLLM_IMAGE,
        network_config: QSRNetworkConfig | None = None,
    ) -> bool:
        if image == VLLM_IMAGE:
            return DockerImageManager.ensure_qwen_asr_image(
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
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int = MAX_NUM_SEQS,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        skip_mm_profiling: bool = DEFAULT_SKIP_MM_PROFILING,
        enable_sleep_mode: bool = DEFAULT_ENABLE_SLEEP_MODE,
        cudagraph_mode: str | None = DEFAULT_CUDAGRAPH_MODE,
        gpu_configs: list[GpuModelConfig] | None = None,
        network_config: QSRNetworkConfig | None = None,
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
        self.network_config = network_config or QSRNetworkConfig.from_overrides(
            proxy=http_proxy,
            hf_endpoint=hf_endpoint,
            pip_index_url=pip_index_url,
            pip_trusted_host=pip_trusted_host,
        )
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.skip_mm_profiling = skip_mm_profiling
        self.enable_sleep_mode = enable_sleep_mode
        self.cudagraph_mode = cudagraph_mode.upper() if cudagraph_mode else None
        self.compilation_config = build_compilation_config(self.cudagraph_mode)
        self.gpu_configs = gpu_configs or []
        self._gpu_config_map = {config.gpu_id: config for config in self.gpu_configs}
        self.model_config_manager = ModelConfigManager(cache_hf_hub=self.cache_hf_hub)

    def _get_gpu_config(self, gpu: GPUInfo) -> GpuModelConfig:
        if gpu.index in self._gpu_config_map:
            return self._gpu_config_map[gpu.index]
        return GpuModelConfig(gpu_id=gpu.index, model_name=self.model_name)

    def _uses_runtime_offline_cache(self) -> bool:
        model_ids = {self._get_gpu_config(gpu).model_name for gpu in self.gpus}
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
                    "tokenizer_config.json",
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
        lines = ["# QSR (Qwen3-ASR) deployment via vLLM"]
        if self._gpu_config_map:
            lines.append("# Per-GPU configs:")
            for gpu in self.gpus:
                config = self._get_gpu_config(gpu)
                lines.append(f"#   GPU {config.gpu_id}: {config.display_label}")
        else:
            lines.append(f"# Model: {self.model_name}")

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
            lines.extend(["    - HF_HUB_OFFLINE=1", "    - TRANSFORMERS_OFFLINE=1"])

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
            ["  ipc: host", "  ulimits:", "    memlock: -1", "    stack: 67108864", ""]
        )
        return lines

    def _generate_service(self, gpu: GPUInfo) -> list[str]:
        gpu_config = self._get_gpu_config(gpu)
        service_port = self.port + gpu.index
        container_name = f"{self.project_name}--gpu{gpu.index}"
        serve_args = [
            "vllm",
            "serve",
            gpu_config.model_name,
            "--served-model-name",
            gpu_config.served_model_name,
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--dtype",
            "half",
            "--trust-remote-code",
        ]

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

        if self.hf_token:
            serve_args.extend(["--api-key", self.hf_token])

        lines = [
            f"  qsr-gpu{gpu.index}:",
            "    <<: *common-config",
            f"    image: {gpu.image}",
            f"    container_name: {container_name}",
            "    ports:",
            f'      - "{service_port}:{VLLM_INTERNAL_PORT}"',
            "    entrypoint:",
            "      - /bin/bash",
            "      - -lc",
            "    command:",
            f"      - exec {' '.join(shlex.quote(arg) for arg in serve_args)}",
        ]

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


class QSRComposer:
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
        max_model_len: int = MAX_MODEL_LEN,
        max_num_seqs: int = MAX_NUM_SEQS,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        skip_mm_profiling: bool = DEFAULT_SKIP_MM_PROFILING,
        enable_sleep_mode: bool = DEFAULT_ENABLE_SLEEP_MODE,
        cudagraph_mode: str | None = DEFAULT_CUDAGRAPH_MODE,
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
        self.network_config = QSRNetworkConfig.from_overrides(
            proxy=http_proxy,
            hf_endpoint=hf_endpoint,
            pip_index_url=pip_index_url,
            pip_trusted_host=pip_trusted_host,
        )
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.skip_mm_profiling = skip_mm_profiling
        self.enable_sleep_mode = enable_sleep_mode
        self.cudagraph_mode = cudagraph_mode.upper() if cudagraph_mode else None
        self.gpu_layout = normalize_model_key(gpu_layout) if gpu_layout else None

        if self.gpu_layout:
            project_dash = re.sub(r"[^a-z0-9_-]", "-", self.gpu_layout)
            self.project_name = project_name or f"qsr-{project_dash}"
        elif gpu_configs:
            self.project_name = project_name or "qsr-multi"
        else:
            project_dash = self.model_name.replace("/", "--").lower()
            project_dash = re.sub(r"[^a-z0-9_-]", "_", project_dash)
            self.project_name = project_name or f"qsr--{project_dash}"

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

    def _get_discovery_project_name(self) -> str:
        compose_file = self._find_compose_file()
        if compose_file is not None:
            return compose_file.stem
        return self.project_name

    def _find_compose_file(self) -> Path | None:
        if self.compose_file.exists():
            return self.compose_file

        multi = self.compose_dir / "qsr-multi.yml"
        if multi.exists():
            return multi

        candidates = sorted(self.compose_dir.glob("qsr*.yml"))
        if candidates:
            return candidates[0]
        return None

    @staticmethod
    def _find_qsr_containers(include_stopped: bool = False) -> list[str]:
        flag = "-a" if include_stopped else ""
        result = subprocess.run(
            f'docker ps {flag} --filter "name=qsr" --format "{{{{.Names}}}}"',
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        names = [
            name.strip()
            for name in result.stdout.splitlines()
            if name.strip() and re.match(r"qsr[-_]", name.strip())
        ]
        return sorted(names)

    def _resolve_compose_cmd(self, cmd: str) -> bool:
        compose_file = self._find_compose_file()
        if not compose_file:
            return False
        if compose_file != self.compose_file:
            logger.mesg(f"[qsr] Using compose file: {compose_file}")
        full_cmd = f"docker compose -f {compose_file} {cmd}"
        logger.mesg(f"[qsr] Running: {full_cmd}")
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
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            skip_mm_profiling=self.skip_mm_profiling,
            enable_sleep_mode=self.enable_sleep_mode,
            cudagraph_mode=self.cudagraph_mode,
            gpu_configs=self.gpu_configs,
            network_config=self.network_config,
        )
        self.compose_file.write_text(generator.generate())
        logger.okay(f"[qsr] Generated: {self.compose_file}")
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

    def _discover_running_backend_targets(self) -> list[tuple[str, str]]:
        result = subprocess.run(
            "docker ps --format '{{.Names}}|{{.Ports}}'",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        prefix = f"{self._get_discovery_project_name()}--gpu"
        targets: list[tuple[str, str]] = []
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
            targets.append((container_name, f"http://localhost:{host_port}"))

        targets.sort()
        return targets

    def _discover_running_backend_endpoints(self) -> list[str]:
        return [endpoint for _, endpoint in self._discover_running_backend_targets()]

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

        def request_endpoint(endpoint: str) -> tuple[str, str, dict | str | None]:
            with httpx.Client(
                timeout=httpx.Timeout(SLEEP_CONTROL_TIMEOUT_SEC)
            ) as client:
                url = f"{endpoint}{path}"
                try:
                    response = client.request(method, url, params=params)
                except Exception as exc:
                    return endpoint, "failure", str(exc)

                if response.status_code == 200:
                    payload = None
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type.lower():
                        try:
                            payload = response.json()
                        except ValueError:
                            payload = None
                    return endpoint, "success", payload
                if response.status_code == 404:
                    return endpoint, "unsupported", None
                return endpoint, "failure", f"HTTP {response.status_code}"

        with ThreadPoolExecutor(max_workers=max(1, len(endpoints))) as executor:
            futures = [
                executor.submit(request_endpoint, endpoint) for endpoint in endpoints
            ]
            for future in as_completed(futures):
                endpoint, status, payload = future.result()
                if status == "success":
                    successes.append(
                        (endpoint, payload if isinstance(payload, dict) else None)
                    )
                elif status == "unsupported":
                    unsupported.append(endpoint)
                else:
                    failures.append((endpoint, str(payload)))

        successes.sort(key=lambda item: item[0])
        unsupported.sort()
        failures.sort(key=lambda item: item[0])

        return successes, unsupported, failures

    @staticmethod
    def _build_embedded_warmup_audio() -> tuple[str, bytes, str]:
        sample_rate = 16000
        duration_sec = 0.25
        frequency_hz = 440.0
        amplitude = 0.2
        frame_count = max(1, int(sample_rate * duration_sec))

        pcm = bytearray()
        for index in range(frame_count):
            sample = math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
            sample_int = int(32767 * amplitude * sample)
            pcm.extend(sample_int.to_bytes(2, byteorder="little", signed=True))

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(pcm))

        return "qsr-warmup.wav", buffer.getvalue(), "audio/wav"

    @staticmethod
    def _load_warmup_audio(audio: str | None) -> tuple[str, bytes, str]:
        if not audio:
            return QSRComposer._build_embedded_warmup_audio()

        audio_path = Path(audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Warmup audio not found: {audio}")

        mime_type, _ = mimetypes.guess_type(audio_path.name)
        return (
            audio_path.name,
            audio_path.read_bytes(),
            mime_type or "application/octet-stream",
        )

    @staticmethod
    def _resolve_warmup_model(
        client: httpx.Client,
        endpoint: str,
        fallback_model: str,
    ) -> str:
        for route in ("/v1/models", "/models"):
            response = client.get(f"{endpoint}{route}")
            if response.status_code in (404, 405):
                continue
            response.raise_for_status()

            payload = response.json()
            for item in payload.get("data", []):
                model_id = str(item.get("id", "")).strip()
                if model_id:
                    return model_id
            break
        return fallback_model

    def _warmup_endpoint(
        self,
        endpoint: str,
        *,
        audio_upload: tuple[str, bytes, str],
        request_timeout_sec: float,
    ) -> tuple[bool, str]:
        filename, payload, mime_type = audio_upload
        fallback_model = get_model_shortcut(self.model_name) or self.model_name
        client = httpx.Client(timeout=httpx.Timeout(request_timeout_sec))
        try:
            model_name = self._resolve_warmup_model(client, endpoint, fallback_model)
            files: list[tuple[str, object]] = []
            if model_name:
                files.append(("model", (None, model_name)))
            files.append(("response_format", (None, "text")))
            files.append(("file", (filename, payload, mime_type)))

            last_error: Exception | None = None
            for route in ("/v1/audio/transcriptions", "/audio/transcriptions"):
                try:
                    response = client.post(f"{endpoint}{route}", files=files)
                    if response.status_code in (404, 405):
                        continue
                    response.raise_for_status()
                    preview = " ".join(response.text.strip().split())
                    if len(preview) > 96:
                        preview = preview[:93] + "..."
                    return True, preview or "ok"
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in (404, 405):
                        last_error = exc
                        continue
                    detail = exc.response.text.strip() or str(exc)
                    return False, detail
                except Exception as exc:
                    return False, str(exc)

            if last_error is not None:
                return False, str(last_error)
            return False, "No transcription endpoint available"
        except Exception as exc:
            return False, str(exc)
        finally:
            client.close()

    def wait_for_healthy_backends(
        self,
        timeout_sec: float = 300.0,
        poll_interval_sec: float = 5.0,
        label: str = "[qsr]",
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

    def wait_for_ready_backends(
        self,
        endpoints: list[str] | None = None,
        timeout_sec: float = WARMUP_WAIT_TIMEOUT_SEC,
        poll_interval_sec: float = WARMUP_POLL_INTERVAL_SEC,
        request_timeout_sec: float = READINESS_REQUEST_TIMEOUT_SEC,
        label: str = "[qsr]",
    ) -> bool:
        target_endpoints = [
            endpoint.rstrip("/") for endpoint in (endpoints or []) if endpoint
        ] or self.get_backend_endpoints()
        healthy = wait_for_healthy_http_endpoints(
            target_endpoints,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            request_timeout_sec=request_timeout_sec,
            label=label,
        )
        if healthy:
            set_backend_sleep_states(target_endpoints, sleeping=False)
        return healthy

    def _profile_container_health(
        self,
        *,
        timeout_sec: float,
        poll_interval_sec: float,
        label: str,
    ) -> tuple[bool, dict[str, float]]:
        container_names = self.get_backend_container_names()
        if not container_names:
            logger.warn(
                f"× {label} No backend containers available for startup profile"
            )
            return False, {}

        pending = set(container_names)
        timings: dict[str, float] = {}
        started_at = time.monotonic()
        deadline = time.monotonic() + timeout_sec

        logger.mesg(
            f"{label} Profiling {len(pending)} backend container(s) until Docker health is ready"
        )
        last_statuses: dict[str, str] = {}
        while pending and time.monotonic() < deadline:
            last_statuses = get_docker_container_statuses(container_names)
            now = time.monotonic()
            for container_name in list(pending):
                if (
                    docker_status_to_health(last_statuses.get(container_name, ""))
                    is True
                ):
                    timings[container_name] = now - started_at
                    pending.remove(container_name)

            if pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(poll_interval_sec, remaining))

        if pending:
            missing = []
            for container_name in sorted(pending):
                status = last_statuses.get(container_name)
                if status:
                    missing.append(f"{container_name} ({status})")
                else:
                    missing.append(container_name)
            logger.warn(
                f"× {label} Timed out profiling healthy containers: {', '.join(missing)}"
            )
            return False, timings

        return True, timings

    def _profile_endpoint_readiness(
        self,
        targets: list[tuple[str, str]],
        *,
        timeout_sec: float,
        poll_interval_sec: float,
        request_timeout_sec: float,
        label: str,
    ) -> tuple[bool, dict[str, float]]:
        if not targets:
            logger.warn(f"× {label} No backend endpoints available for startup profile")
            return False, {}

        pending = {
            container_name: endpoint.rstrip("/")
            for container_name, endpoint in targets
            if endpoint
        }
        timings: dict[str, float] = {}
        started_at = time.monotonic()
        deadline = time.monotonic() + timeout_sec

        logger.mesg(
            f"{label} Profiling {len(pending)} backend endpoint(s) until HTTP readiness"
        )

        def probe(container_name: str, endpoint: str) -> tuple[str, bool]:
            try:
                response = httpx.get(
                    f"{endpoint}/health",
                    timeout=httpx.Timeout(request_timeout_sec),
                )
                return container_name, response.status_code == 200
            except Exception:
                return container_name, False

        max_workers = max(1, min(32, len(pending)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending and time.monotonic() < deadline:
                future_to_name = {
                    executor.submit(probe, container_name, endpoint): container_name
                    for container_name, endpoint in pending.items()
                }
                for future in as_completed(future_to_name):
                    container_name, healthy = future.result()
                    if healthy and container_name in pending:
                        timings[container_name] = time.monotonic() - started_at
                        del pending[container_name]

                if pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(poll_interval_sec, remaining))

        if pending:
            missing = ", ".join(sorted(pending))
            logger.warn(f"× {label} Timed out profiling ready endpoints: {missing}")
            return False, timings

        return True, timings

    def _run_warmup_targets(
        self,
        targets: list[tuple[str, str]],
        *,
        audio_upload: tuple[str, bytes, str],
        request_timeout_sec: float,
    ) -> tuple[dict[str, tuple[bool, str]], dict[str, float]]:
        results: dict[str, tuple[bool, str]] = {}
        durations: dict[str, float] = {}

        def warm(container_name: str, endpoint: str) -> tuple[str, bool, str, float]:
            started_at = time.perf_counter()
            okay, message = self._warmup_endpoint(
                endpoint,
                audio_upload=audio_upload,
                request_timeout_sec=request_timeout_sec,
            )
            return container_name, okay, message, time.perf_counter() - started_at

        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            future_to_name = {
                executor.submit(warm, container_name, endpoint): container_name
                for container_name, endpoint in targets
            }
            for future in as_completed(future_to_name):
                container_name, okay, message, duration = future.result()
                results[container_name] = (okay, message)
                durations[container_name] = duration

        return results, durations

    @staticmethod
    def _log_warmup_results(
        targets: list[tuple[str, str]],
        results: dict[str, tuple[bool, str]],
    ) -> list[str]:
        failures: list[str] = []
        for container_name, _endpoint in targets:
            okay, message = results.get(
                container_name,
                (False, "warmup task did not complete"),
            )
            if okay:
                logger.okay(f"[qsr] Warmed {container_name}: {message}")
            else:
                logger.warn(f"[qsr] Warmup failed for {container_name}: {message}")
                failures.append(container_name)
        return failures

    def _run_compose_cmd(
        self,
        cmd: str,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        full_cmd = f"docker compose -f {self.compose_file} {cmd}"
        logger.mesg(f"[qsr] Running: {full_cmd}")
        return subprocess.run(full_cmd, shell=True, capture_output=capture_output)

    def _profile_startup(
        self,
        *,
        warmup_audio: str | None,
        wait_timeout_sec: float,
        poll_interval_sec: float,
        request_timeout_sec: float,
        profile: StartupProfile,
    ) -> None:
        container_started_at = time.perf_counter()
        container_healthy, container_timings = self._profile_container_health(
            timeout_sec=wait_timeout_sec,
            poll_interval_sec=poll_interval_sec,
            label="[qsr]",
        )
        profile.container_health_elapsed_sec = (
            time.perf_counter() - container_started_at
        )
        profile.container_health_by_container = container_timings

        targets = self._discover_running_backend_targets()
        profile.endpoint_by_container = {
            container_name: endpoint for container_name, endpoint in targets
        }
        if not targets:
            logger.warn("[qsr] Startup profile could not find any running backends")
            profile.total_elapsed_sec = (
                profile.image_ready_elapsed_sec
                + profile.compose_generate_elapsed_sec
                + profile.compose_up_elapsed_sec
                + (profile.container_health_elapsed_sec or 0.0)
            )
            profile.log()
            return

        endpoint_started_at = time.perf_counter()
        ready, endpoint_timings = self._profile_endpoint_readiness(
            targets,
            timeout_sec=wait_timeout_sec,
            poll_interval_sec=poll_interval_sec,
            request_timeout_sec=READINESS_REQUEST_TIMEOUT_SEC,
            label="[qsr]",
        )
        profile.endpoint_ready_elapsed_sec = time.perf_counter() - endpoint_started_at
        profile.endpoint_ready_by_container = endpoint_timings
        if endpoint_timings:
            set_backend_sleep_states(
                [endpoint for _, endpoint in targets], sleeping=False
            )

        if not container_healthy or not ready:
            profile.total_elapsed_sec = (
                profile.image_ready_elapsed_sec
                + profile.compose_generate_elapsed_sec
                + profile.compose_up_elapsed_sec
                + (profile.container_health_elapsed_sec or 0.0)
                + (profile.endpoint_ready_elapsed_sec or 0.0)
            )
            profile.log()
            return

        audio_upload = self._load_warmup_audio(warmup_audio)
        warmup_started_at = time.perf_counter()
        results, durations = self._run_warmup_targets(
            targets,
            audio_upload=audio_upload,
            request_timeout_sec=request_timeout_sec,
        )
        profile.warmup_elapsed_sec = time.perf_counter() - warmup_started_at
        profile.warmup_by_container = durations
        profile.warmup_result_by_container = {
            container_name: message
            for container_name, (_okay, message) in results.items()
        }

        failures = self._log_warmup_results(targets, results)
        if failures:
            logger.warn(f"[qsr] Warmup completed with failures: {', '.join(failures)}")
        else:
            logger.okay(f"[qsr] Warmed {len(targets)} backend(s)")
            set_backend_sleep_states(
                [endpoint for _, endpoint in targets], sleeping=False
            )

        profile.total_elapsed_sec = (
            profile.image_ready_elapsed_sec
            + profile.compose_generate_elapsed_sec
            + profile.compose_up_elapsed_sec
            + (profile.container_health_elapsed_sec or 0.0)
            + (profile.endpoint_ready_elapsed_sec or 0.0)
            + (profile.warmup_elapsed_sec or 0.0)
        )
        profile.log()

    def up(
        self,
        *,
        skip_warmup: bool = False,
        warmup_audio: str | None = None,
        wait_timeout_sec: float = WARMUP_WAIT_TIMEOUT_SEC,
        poll_interval_sec: float = WARMUP_POLL_INTERVAL_SEC,
        request_timeout_sec: float = WARMUP_REQUEST_TIMEOUT_SEC,
        profile_startup: bool = False,
    ) -> None:
        if not self.gpus:
            logger.warn("× No healthy GPUs detected")
            summary = GPUDetector.get_unhealthy_gpu_summary()
            if summary:
                logger.warn(summary)
            return

        logger.mesg(f"[qsr] Starting vLLM for model: {self.model_name}")
        if self.gpu_configs:
            for config in self.gpu_configs:
                logger.mesg(f"[qsr]   GPU {config.gpu_id}: {config.display_label}")

        profile = StartupProfile()

        image_started_at = time.perf_counter()
        if not self.image_manager.ensure_image(
            VLLM_IMAGE, network_config=self.network_config
        ):
            return
        profile.image_ready_elapsed_sec = time.perf_counter() - image_started_at

        self._ensure_data_dir()
        generate_started_at = time.perf_counter()
        self.generate_compose_file()
        profile.compose_generate_elapsed_sec = time.perf_counter() - generate_started_at
        compose_up_started_at = time.perf_counter()
        result = self._run_compose_cmd("up -d --remove-orphans", capture_output=True)
        profile.compose_up_elapsed_sec = time.perf_counter() - compose_up_started_at
        if result.returncode != 0:
            stderr = (
                result.stderr.decode("utf-8", errors="replace")
                if isinstance(result.stderr, bytes)
                else result.stderr
            )
            logger.warn(f"[qsr] Startup failed: {stderr.strip()}")
            return
        logger.okay(
            f"[qsr] Started services on GPUs: {[gpu.index for gpu in self.gpus]}"
        )
        set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)
        self.ps()
        if profile_startup:
            if skip_warmup:
                profile.total_elapsed_sec = (
                    profile.image_ready_elapsed_sec
                    + profile.compose_generate_elapsed_sec
                    + profile.compose_up_elapsed_sec
                )
                logger.note(
                    "[qsr] Startup profiling requested with --skip-warmup; reporting compose-only timings"
                )
                profile.log()
                return
            self._profile_startup(
                warmup_audio=warmup_audio,
                wait_timeout_sec=wait_timeout_sec,
                poll_interval_sec=poll_interval_sec,
                request_timeout_sec=request_timeout_sec,
                profile=profile,
            )
            return
        if skip_warmup:
            return
        self.warmup(
            audio=warmup_audio,
            wait_timeout_sec=wait_timeout_sec,
            poll_interval_sec=poll_interval_sec,
            request_timeout_sec=request_timeout_sec,
        )

    def down(self) -> None:
        if self._resolve_compose_cmd("down --remove-orphans"):
            clear_backend_sleep_states(self.get_backend_endpoints())
            return

        containers = self._find_qsr_containers(include_stopped=True)
        if not containers:
            logger.mesg("[qsr] No QSR containers found")
            return
        logger.mesg(f"[qsr] Removing {len(containers)} container(s)")
        for container in containers:
            subprocess.run(f"docker rm -f {container}", shell=True, capture_output=True)
        logger.okay(f"[qsr] Removed {len(containers)} container(s)")
        clear_backend_sleep_states(self.get_backend_endpoints())

    def stop(self) -> None:
        if self._resolve_compose_cmd("stop"):
            return

        containers = self._find_qsr_containers(include_stopped=False)
        if not containers:
            logger.mesg("[qsr] No running QSR containers found")
            return
        for container in containers:
            subprocess.run(f"docker stop {container}", shell=True, capture_output=True)
        logger.okay(f"[qsr] Stopped {len(containers)} container(s)")

    def start(self) -> None:
        if self._resolve_compose_cmd("start"):
            set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)
            return

        containers = self._find_qsr_containers(include_stopped=True)
        running = set(self._find_qsr_containers(include_stopped=False))
        stopped = [container for container in containers if container not in running]
        if not stopped:
            logger.mesg("[qsr] No stopped QSR containers found")
            return
        for container in stopped:
            subprocess.run(f"docker start {container}", shell=True, capture_output=True)
        logger.okay(f"[qsr] Started {len(stopped)} container(s)")
        set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)

    def restart(self) -> None:
        if self._resolve_compose_cmd("restart"):
            set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)
            return

        containers = self._find_qsr_containers(include_stopped=False)
        if not containers:
            logger.mesg("[qsr] No running QSR containers found")
            return
        for container in containers:
            subprocess.run(
                f"docker restart {container}", shell=True, capture_output=True
            )
        logger.okay(f"[qsr] Restarted {len(containers)} container(s)")
        set_backend_sleep_states(self.get_backend_endpoints(), sleeping=False)

    def warmup(
        self,
        audio: str | None = None,
        wait_timeout_sec: float = WARMUP_WAIT_TIMEOUT_SEC,
        poll_interval_sec: float = WARMUP_POLL_INTERVAL_SEC,
        request_timeout_sec: float = WARMUP_REQUEST_TIMEOUT_SEC,
        wait_for_ready: bool = True,
    ) -> None:
        targets = self._discover_running_backend_targets()
        if not targets:
            logger.warn("[qsr] No running QSR backends found for warmup")
            return

        if wait_for_ready:
            logger.mesg(f"[qsr] Waiting for {len(targets)} backend(s) before warmup")
            if not self.wait_for_ready_backends(
                endpoints=[endpoint for _, endpoint in targets],
                timeout_sec=wait_timeout_sec,
                poll_interval_sec=poll_interval_sec,
                request_timeout_sec=READINESS_REQUEST_TIMEOUT_SEC,
                label="[qsr]",
            ):
                logger.warn(
                    "[qsr] Warmup aborted: backends did not become healthy in time"
                )
                return

        audio_upload = self._load_warmup_audio(audio)
        results, _durations = self._run_warmup_targets(
            targets,
            audio_upload=audio_upload,
            request_timeout_sec=request_timeout_sec,
        )
        failures = self._log_warmup_results(targets, results)

        if failures:
            logger.warn(f"[qsr] Warmup completed with failures: {', '.join(failures)}")
            return
        logger.okay(f"[qsr] Warmed {len(targets)} backend(s)")
        set_backend_sleep_states([endpoint for _, endpoint in targets], sleeping=False)

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
                [endpoint for endpoint, _ in successes],
                sleeping=True,
            )
            logger.okay(
                f"[qsr] Requested sleep(level={level}, mode={mode}) on {len(successes)} backend(s)"
            )
        elif unsupported:
            logger.warn(
                "[qsr] Sleep endpoints are unavailable. Redeploy with --enable-sleep-mode to use fast wake/resume."
            )
        else:
            logger.warn("[qsr] Failed to contact any running backend for sleep")

        for endpoint, error in failures:
            logger.warn(f"[qsr] Sleep request failed for {endpoint}: {error}")
        return bool(successes)

    def wake(self, wait_healthy: bool = False) -> bool:
        successes, unsupported, failures = self._request_backend_control("/wake_up")

        if successes:
            set_backend_sleep_states(
                [endpoint for endpoint, _ in successes],
                sleeping=False,
            )
            logger.okay(f"[qsr] Requested wake-up on {len(successes)} backend(s)")
        elif unsupported:
            logger.warn(
                "[qsr] Wake-up endpoints are unavailable. Redeploy with --enable-sleep-mode to use fast wake/resume."
            )
        else:
            logger.warn("[qsr] Failed to contact any running backend for wake-up")

        for endpoint, error in failures:
            logger.warn(f"[qsr] Wake request failed for {endpoint}: {error}")

        if wait_healthy and successes:
            healthy = self.wait_for_healthy_backends(
                timeout_sec=SLEEP_WAKE_TIMEOUT_SEC,
                poll_interval_sec=SLEEP_WAKE_POLL_INTERVAL_SEC,
                label="[qsr]",
            )
            if healthy:
                logger.okay("[qsr] Backends are healthy after wake-up")
            else:
                logger.warn(
                    "[qsr] Wake-up requested, but healthy backends did not appear in time"
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
                "[qsr] Sleep status endpoints are unavailable. Redeploy with --enable-sleep-mode to inspect sleep state."
            )
            return False

        if not successes and not failures:
            logger.warn("[qsr] No running backends found for sleep status")
            return False

        for endpoint, payload in successes:
            sleeping = bool((payload or {}).get("is_sleeping", False))
            update_backend_sleep_states({endpoint: sleeping})
            state = "sleeping" if sleeping else "awake"
            logger.mesg(f"[qsr] {endpoint}: {state}")

        for endpoint, error in failures:
            logger.warn(f"[qsr] Sleep status failed for {endpoint}: {error}")

        return bool(successes)

    def ps(self) -> None:
        if self._resolve_compose_cmd("ps"):
            return
        self._show_manual_status()

    def logs(self, follow: bool = False, tail: int = 100) -> None:
        follow_flag = "-f" if follow else ""
        cmd = f"logs --tail={tail} {follow_flag}".strip()
        if self._resolve_compose_cmd(cmd):
            return

        containers = self._find_qsr_containers(include_stopped=True)
        if not containers:
            logger.mesg("[qsr] No QSR containers found")
            return
        for container in containers:
            logger.mesg(f"[qsr] Logs for {container}:")
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
  qsr compose up
  qsr compose up -g 0,1
  qsr compose up --gpu-layout uniform
    qsr compose warmup --gpu-layout uniform
    qsr compose wake --wait-healthy
    qsr compose sleep --sleep-level 1 --sleep-mode abort
    qsr compose sleep-status
  qsr compose up --gpu-configs "0,1"
  qsr compose up --gpu-configs "0:Qwen/Qwen3-ASR-0.6B,1:Qwen/Qwen3-ASR-0.6B"
  qsr compose generate -j qsr-demo --gpu-configs "0"
  qsr compose logs -f
  qsr compose down
"""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Manage QSR vLLM Docker compose deployments"
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
            help='Per-GPU config: "GPU[:MODEL],..."',
        )
        target.add_argument(
            "--gpu-layout",
            choices=[GPU_LAYOUT_UNIFORM],
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
        target.set_defaults(enable_sleep_mode=DEFAULT_ENABLE_SLEEP_MODE)
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
        target.add_argument(
            "--enable-sleep-mode",
            dest="enable_sleep_mode",
            action="store_true",
            help="Enable vLLM sleep-mode endpoints so warm backends can wake faster than cold start",
        )
        target.add_argument(
            "--no-enable-sleep-mode",
            dest="enable_sleep_mode",
            action="store_false",
            help="Disable vLLM sleep-mode endpoints",
        )
        target.add_argument(
            "--cudagraph-mode",
            choices=CUDAGRAPH_MODE_CHOICES,
            default=DEFAULT_CUDAGRAPH_MODE,
            help=(
                "vLLM cudagraph startup mode override. Default NONE prefers faster "
                "readiness for QSR cold starts."
            ),
        )
        target.add_argument(
            "--skip-warmup",
            action="store_true",
            help="Start containers without waiting for readiness and default warmup",
        )
        target.add_argument(
            "--warmup-audio",
            type=str,
            default=None,
            help="Optional local audio file to use for the default warmup probe",
        )
        target.add_argument(
            "--wait-timeout",
            type=float,
            default=WARMUP_WAIT_TIMEOUT_SEC,
            help=f"Seconds to wait for backend readiness (default: {WARMUP_WAIT_TIMEOUT_SEC})",
        )
        target.add_argument(
            "--poll-interval",
            type=float,
            default=WARMUP_POLL_INTERVAL_SEC,
            help=f"Backend readiness polling interval in seconds (default: {WARMUP_POLL_INTERVAL_SEC})",
        )
        target.add_argument(
            "--request-timeout",
            type=float,
            default=WARMUP_REQUEST_TIMEOUT_SEC,
            help=f"Per-backend warmup request timeout in seconds (default: {WARMUP_REQUEST_TIMEOUT_SEC})",
        )
        target.add_argument(
            "--profile-startup",
            action="store_true",
            help="Measure and report cold-start phases including Docker health, HTTP readiness, and warmup",
        )
        add_targeting_arguments(target)

    parser_up = subparsers.add_parser("up", help="Start QSR containers")
    add_common_arguments(parser_up)
    add_deployment_arguments(parser_up)

    parser_generate = subparsers.add_parser("generate", help="Generate compose file")
    add_common_arguments(parser_generate)
    add_deployment_arguments(parser_generate)

    parser_sleep = subparsers.add_parser(
        "sleep",
        help="Put running QSR backends into vLLM sleep mode",
    )
    add_common_arguments(parser_sleep)
    add_targeting_arguments(parser_sleep)
    parser_sleep.add_argument(
        "--sleep-level",
        type=int,
        default=DEFAULT_SLEEP_LEVEL,
        help=f"vLLM sleep level (default: {DEFAULT_SLEEP_LEVEL})",
    )
    parser_sleep.add_argument(
        "--sleep-mode",
        type=str,
        default=DEFAULT_SLEEP_MODE,
        help=f"vLLM sleep mode (default: {DEFAULT_SLEEP_MODE})",
    )

    parser_wake = subparsers.add_parser(
        "wake",
        help="Wake running QSR backends from vLLM sleep mode",
    )
    add_common_arguments(parser_wake)
    add_targeting_arguments(parser_wake)
    parser_wake.add_argument(
        "--wait-healthy",
        action="store_true",
        help="Wait for backends to report healthy after wake-up",
    )

    parser_sleep_status = subparsers.add_parser(
        "sleep-status",
        help="Show whether running QSR backends are sleeping",
    )
    add_common_arguments(parser_sleep_status)
    add_targeting_arguments(parser_sleep_status)

    parser_warmup = subparsers.add_parser(
        "warmup",
        help="Warm running QSR backends with a short transcription probe",
    )
    add_common_arguments(parser_warmup)
    add_targeting_arguments(parser_warmup)
    parser_warmup.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Optional local audio file for warmup; uses an embedded WAV when omitted",
    )
    parser_warmup.add_argument(
        "--wait-timeout",
        type=float,
        default=WARMUP_WAIT_TIMEOUT_SEC,
        help=f"Seconds to wait for healthy backends (default: {WARMUP_WAIT_TIMEOUT_SEC})",
    )
    parser_warmup.add_argument(
        "--poll-interval",
        type=float,
        default=WARMUP_POLL_INTERVAL_SEC,
        help=f"Health polling interval in seconds (default: {WARMUP_POLL_INTERVAL_SEC})",
    )
    parser_warmup.add_argument(
        "--request-timeout",
        type=float,
        default=WARMUP_REQUEST_TIMEOUT_SEC,
        help=f"Per-backend warmup request timeout in seconds (default: {WARMUP_REQUEST_TIMEOUT_SEC})",
    )

    for action in ("down", "stop", "start", "restart", "ps", "health"):
        action_parser = subparsers.add_parser(
            action, help=f"{action.title()} QSR containers"
        )
        add_common_arguments(action_parser)

    parser_logs = subparsers.add_parser("logs", help="Show QSR logs")
    add_common_arguments(parser_logs)
    parser_logs.add_argument("-f", "--follow", action="store_true")
    parser_logs.add_argument("--tail", type=int, default=100)


def run_from_args(args: argparse.Namespace) -> None:
    gpu_configs = None
    if getattr(args, "gpu_configs", None):
        gpu_configs = parse_gpu_configs(args.gpu_configs)

    composer = QSRComposer(
        model_name=getattr(args, "model_name", MODEL_NAME),
        port=getattr(args, "port", SERVER_PORT),
        project_name=getattr(args, "project_name", None),
        gpu_ids=getattr(args, "gpus", None),
        hf_token=getattr(args, "hf_token", None),
        mount_mode=getattr(args, "mount_mode", DEVICE_MOUNT_MODE),
        http_proxy=getattr(args, "proxy", None),
        hf_endpoint=getattr(args, "hf_endpoint", None),
        pip_index_url=getattr(args, "pip_index_url", None),
        pip_trusted_host=getattr(args, "pip_trusted_host", None),
        max_model_len=getattr(args, "max_model_len", MAX_MODEL_LEN),
        max_num_seqs=getattr(args, "max_num_seqs", MAX_NUM_SEQS),
        gpu_memory_utilization=getattr(
            args,
            "gpu_memory_utilization",
            GPU_MEMORY_UTILIZATION,
        ),
        skip_mm_profiling=getattr(
            args,
            "skip_mm_profiling",
            DEFAULT_SKIP_MM_PROFILING,
        ),
        enable_sleep_mode=getattr(
            args,
            "enable_sleep_mode",
            DEFAULT_ENABLE_SLEEP_MODE,
        ),
        cudagraph_mode=getattr(
            args,
            "cudagraph_mode",
            DEFAULT_CUDAGRAPH_MODE,
        ),
        gpu_layout=getattr(args, "gpu_layout", None),
        gpu_configs=gpu_configs,
    )

    action = args.compose_action
    if action == "generate":
        composer.generate_compose_file()
    elif action == "up":
        composer.up(
            skip_warmup=getattr(args, "skip_warmup", False),
            warmup_audio=getattr(args, "warmup_audio", None),
            wait_timeout_sec=getattr(args, "wait_timeout", WARMUP_WAIT_TIMEOUT_SEC),
            poll_interval_sec=getattr(
                args,
                "poll_interval",
                WARMUP_POLL_INTERVAL_SEC,
            ),
            request_timeout_sec=getattr(
                args,
                "request_timeout",
                WARMUP_REQUEST_TIMEOUT_SEC,
            ),
            profile_startup=getattr(args, "profile_startup", False),
        )
    elif action == "warmup":
        composer.warmup(
            audio=getattr(args, "audio", None),
            wait_timeout_sec=getattr(args, "wait_timeout", WARMUP_WAIT_TIMEOUT_SEC),
            poll_interval_sec=getattr(
                args,
                "poll_interval",
                WARMUP_POLL_INTERVAL_SEC,
            ),
            request_timeout_sec=getattr(
                args,
                "request_timeout",
                WARMUP_REQUEST_TIMEOUT_SEC,
            ),
        )
    elif action == "sleep":
        composer.sleep(
            level=getattr(args, "sleep_level", DEFAULT_SLEEP_LEVEL),
            mode=getattr(args, "sleep_mode", DEFAULT_SLEEP_MODE),
        )
    elif action == "wake":
        composer.wake(wait_healthy=getattr(args, "wait_healthy", False))
    elif action == "sleep-status":
        composer.sleep_status()
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
    elif action == "health":
        composer.health()
    elif action == "logs":
        composer.logs(follow=args.follow, tail=args.tail)
    else:
        raise ValueError(f"Unknown compose action: {action}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
