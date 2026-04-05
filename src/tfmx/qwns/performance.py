"""Performance persistence and tracking for QWN clients."""

import hashlib
import json

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tclogger import logger

from .compose import MAX_CONCURRENT_REQUESTS


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ExplorationConfig:
    CONFIG_FILE = "qwn_clients.config.json"

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or CONFIG_DIR
        self.config_path = self.config_dir / self.CONFIG_FILE
        self._config: dict = {}
        self._load()

    def _load(self) -> None:
        if not self.config_path.exists():
            return
        try:
            self._config = json.loads(self.config_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warn(f"Failed to load QWN exploration config: {exc}")
            self._config = {}

    def _save(self) -> None:
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(self._config, indent=2))
        except OSError as exc:
            logger.warn(f"Failed to save QWN exploration config: {exc}")

    @staticmethod
    def _get_config_key(endpoints: list[str]) -> str:
        key = ",".join(sorted(endpoints))
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @staticmethod
    def _endpoint_to_key(endpoint: str) -> str:
        return endpoint.replace("http://", "").replace("https://", "").rstrip("/")

    def get_machine_config(self, endpoints: list[str], endpoint: str) -> dict | None:
        config_key = self._get_config_key(endpoints)
        config = self._config.get(config_key)
        if not config:
            return None
        machine_key = self._endpoint_to_key(endpoint)
        return config.get("machines", {}).get(machine_key)

    def save_machine_config(
        self,
        endpoints: list[str],
        endpoint: str,
        optimal_max_concurrent: int,
        throughput: float,
        instances: int,
    ) -> None:
        config_key = self._get_config_key(endpoints)
        machine_key = self._endpoint_to_key(endpoint)

        if config_key not in self._config:
            self._config[config_key] = {
                "endpoints": endpoints,
                "machines": {},
            }

        self._config[config_key]["machines"][machine_key] = {
            "optimal_max_concurrent": optimal_max_concurrent,
            "throughput": round(throughput, 1),
            "instances": instances,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()

    def clear(self, endpoints: list[str] | None = None) -> None:
        if endpoints is None:
            self._config = {}
        else:
            self._config.pop(self._get_config_key(endpoints), None)
        self._save()

    def list_configs(self) -> list[dict]:
        configs: list[dict] = []
        for key, value in self._config.items():
            configs.append(
                {
                    "key": key,
                    "endpoints": value.get("endpoints", []),
                    "machines": list(value.get("machines", {}).keys()),
                }
            )
        return configs


@dataclass
class PerformanceMetrics:
    throughput_ema: float = 0.0
    latency_ema: float = 0.0
    ema_alpha: float = 0.3
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    total_latency: float = 0.0

    def update(
        self,
        latency: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> float:
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_latency += latency

        if latency <= 0:
            return 0.0

        throughput = 1.0 / latency
        if self.throughput_ema == 0:
            self.throughput_ema = throughput
            self.latency_ema = latency
        else:
            self.throughput_ema = (
                self.ema_alpha * throughput + (1 - self.ema_alpha) * self.throughput_ema
            )
            self.latency_ema = (
                self.ema_alpha * latency + (1 - self.ema_alpha) * self.latency_ema
            )

        return throughput

    def get_cumulative_throughput(self, elapsed_time: float = 0.0) -> float:
        if elapsed_time > 0:
            return self.total_requests / elapsed_time
        if self.total_latency > 0:
            return self.total_requests / self.total_latency
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency <= 0:
            return 0.0
        return self.total_completion_tokens / self.total_latency


class PerformanceTracker:
    def __init__(self, n_instances: int, saved_config: dict | None = None):
        self.metrics = PerformanceMetrics()
        self.n_instances = n_instances
        self.optimal_max_concurrent = max(n_instances * 2, MAX_CONCURRENT_REQUESTS)
        if saved_config:
            self._load_from_config(saved_config)

    def _load_from_config(self, config: dict) -> bool:
        saved_max_concurrent = config.get("optimal_max_concurrent", 0)
        saved_throughput = config.get("throughput", 0.0)

        if saved_max_concurrent > 0:
            self.optimal_max_concurrent = saved_max_concurrent
        if saved_throughput > 0:
            self.metrics.throughput_ema = saved_throughput
        return True

    def record_request(
        self,
        latency: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.metrics.update(latency, prompt_tokens, completion_tokens)

    def get_stats_dict(self, elapsed_time: float = 0.0) -> dict:
        return {
            "optimal_max_concurrent": self.optimal_max_concurrent,
            "throughput_ema": round(self.metrics.throughput_ema, 3),
            "throughput_cumulative": round(
                self.metrics.get_cumulative_throughput(elapsed_time),
                3,
            ),
            "tokens_per_second": round(self.metrics.tokens_per_second, 1),
            "total_requests": self.metrics.total_requests,
            "total_prompt_tokens": self.metrics.total_prompt_tokens,
            "total_completion_tokens": self.metrics.total_completion_tokens,
        }
