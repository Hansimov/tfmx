"""TEI Performance Optimization and Benchmarking Support

This module provides performance optimization features for TEI clients,
including batch size exploration, throughput tracking, and configuration persistence.

Separates performance/benchmark concerns from core client functionality.
"""

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tclogger import logger
from typing import Optional

from .tei_compose import MAX_CLIENT_BATCH_SIZE


# Config directory
CONFIG_DIR = Path(__file__).parent


class ExplorationConfig:
    """Manages persistence of exploration results for (batch_size, max_concurrent) optimization.

    Stores optimal configurations per endpoint to avoid re-exploration on each run.
    Config file: <module_dir>/tei_clients.config.json

    Format:
    ```
    {
        "b775a741a567": {
            "endpoints": [ "http://localhost:28800", "http://ai122:28800" ],
            "machines": {
            "ai122:28800": {
                "optimal_batch_size": 1750,
                "optimal_max_concurrent": 10,
                "throughput": 291.7,
                "instances": 7,
                "updated_at": "2026-01-14T07:40:23.804785"
            },
            "localhost:28800": {
                "optimal_batch_size": 4750,
                "optimal_max_concurrent": 2,
                "throughput": 687.9,
                "instances": 2,
                "updated_at": "2026-01-14T07:42:50.538637"
            }
            }
        }
    }
    ```
    """

    CONFIG_FILE = "tei_clients.config.json"

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or CONFIG_DIR
        self.config_path = self.config_dir / self.CONFIG_FILE
        self._config: dict = {}
        self._load()

    def _load(self) -> None:
        """Load config from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warn(f"Failed to load exploration config: {e}")
                self._config = {}

    def _save(self) -> None:
        """Save config to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            logger.warn(f"Failed to save exploration config: {e}")

    @staticmethod
    def _get_config_key(endpoints: list[str]) -> str:
        """Generate a unique key for a set of endpoints."""
        # Sort and hash endpoints for consistent key
        sorted_eps = sorted(endpoints)
        key_str = ",".join(sorted_eps)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    @staticmethod
    def _endpoint_to_key(endpoint: str) -> str:
        """Convert endpoint URL to a simple key."""
        # "http://localhost:28800" -> "localhost:28800"
        return endpoint.replace("http://", "").replace("https://", "").rstrip("/")

    def get_machine_config(self, endpoints: list[str], endpoint: str) -> dict | None:
        """Get saved config for a specific machine.

        Args:
            endpoints: Full list of endpoints (for config key lookup)
            endpoint: Specific endpoint to get config for

        Returns:
            Dict with optimal_batch_size, optimal_max_concurrent, throughput, instances, updated_at
            or None if not found
        """
        config_key = self._get_config_key(endpoints)
        if config_key not in self._config:
            return None

        machine_key = self._endpoint_to_key(endpoint)
        machines = self._config[config_key].get("machines", {})
        return machines.get(machine_key)

    def save_machine_config(
        self,
        endpoints: list[str],
        endpoint: str,
        optimal_batch_size: int,
        optimal_max_concurrent: int,
        throughput: float,
        instances: int,
    ) -> None:
        """Save exploration result for a machine.

        Args:
            endpoints: Full list of endpoints
            endpoint: Specific endpoint
            optimal_batch_size: Discovered optimal batch size
            optimal_max_concurrent: Discovered optimal max concurrent requests
            throughput: Achieved throughput at optimal configuration
            instances: Number of GPU instances
        """
        config_key = self._get_config_key(endpoints)
        machine_key = self._endpoint_to_key(endpoint)

        if config_key not in self._config:
            self._config[config_key] = {
                "endpoints": endpoints,
                "machines": {},
            }

        self._config[config_key]["machines"][machine_key] = {
            "optimal_batch_size": optimal_batch_size,
            "optimal_max_concurrent": optimal_max_concurrent,
            "throughput": round(throughput, 1),
            "instances": instances,
            "updated_at": datetime.now().isoformat(),
        }

        self._save()

    def clear(self, endpoints: list[str] | None = None) -> None:
        """Clear saved config.

        Args:
            endpoints: If provided, only clear config for these endpoints.
                      If None, clear all configs.
        """
        if endpoints is None:
            self._config = {}
        else:
            config_key = self._get_config_key(endpoints)
            if config_key in self._config:
                del self._config[config_key]
        self._save()

    def list_configs(self) -> list[dict]:
        """List all saved configurations."""
        configs = []
        for key, data in self._config.items():
            configs.append(
                {
                    "key": key,
                    "endpoints": data.get("endpoints", []),
                    "machines": list(data.get("machines", {}).keys()),
                }
            )
        return configs


@dataclass
class PerformanceMetrics:
    """Performance metrics for a machine or client.

    Tracks throughput, latency, and request statistics.
    """

    # Throughput tracking (EMA for real-time estimation)
    throughput_ema: float = 0.0  # items/second
    latency_ema: float = 0.0  # seconds per batch
    ema_alpha: float = 0.3  # EMA smoothing factor

    # Statistics
    total_items: int = 0
    total_latency: float = 0.0
    total_requests: int = 0

    def update(self, latency: float, n_items: int) -> float:
        """Update metrics with a new measurement.

        Args:
            latency: Request latency in seconds
            n_items: Number of items processed

        Returns:
            Current throughput (items/second)
        """
        self.total_requests += 1
        self.total_items += n_items
        self.total_latency += latency

        if latency <= 0:
            return 0.0

        current_throughput = n_items / latency

        if self.throughput_ema == 0:
            self.throughput_ema = current_throughput
            self.latency_ema = latency
        else:
            self.throughput_ema = (
                self.ema_alpha * current_throughput
                + (1 - self.ema_alpha) * self.throughput_ema
            )
            self.latency_ema = (
                self.ema_alpha * latency + (1 - self.ema_alpha) * self.latency_ema
            )

        return current_throughput

    def get_cumulative_throughput(self, elapsed_time: float = 0.0) -> float:
        """Calculate cumulative throughput.

        Args:
            elapsed_time: Total elapsed time in seconds. If 0, uses total_latency.

        Returns:
            Cumulative throughput (items/second)
        """
        if elapsed_time > 0:
            return self.total_items / elapsed_time
        elif self.total_latency > 0:
            return self.total_items / self.total_latency
        else:
            return 0.0


@dataclass
class ExplorationState:
    """State for batch size and concurrent request exploration.

    Two-phase exploration:
    - Phase 1: Explore batch_size with fixed max_concurrent
    - Phase 2: Explore max_concurrent with optimal batch_size from Phase 1
    """

    # Exploration control
    exploring: bool = True
    explore_phase: int = 1  # 1 = batch_size, 2 = max_concurrent

    # Current exploration values
    explore_values: list[int] = field(default_factory=list)
    explore_index: int = 0
    explore_results: dict = field(default_factory=dict)  # value -> [throughputs]

    # Exploration parameters
    explore_samples_per_value: int = 3  # Samples to collect per value
    explore_step: int = 250  # Step size for batch_size exploration
    explore_min_value: int = 0  # Minimum value before allowing early stop
    explore_decline_count: int = 0  # Consecutive decline count
    explore_decline_max: int = 3  # Max consecutive declines before stopping

    # Configuration limits
    batch_size_min: int = 500
    batch_size_max: int = 3000
    max_concurrent_min: int = 2
    max_concurrent_max: int = 12

    # Exploration results
    best_throughput: float = 0.0
    phase1_best_batch: int = 0
    phase1_best_throughput: float = 0.0
    optimal_batch_size: int = MAX_CLIENT_BATCH_SIZE
    optimal_max_concurrent: int = 6

    # Machine info
    n_instances: int = 1

    def initialize(self, n_instances: int) -> None:
        """Initialize exploration based on number of GPU instances.

        Args:
            n_instances: Number of healthy GPU instances
        """
        self.n_instances = n_instances
        self.explore_phase = 1
        self.exploring = True
        self._init_phase1_batch_exploration()

    def _init_phase1_batch_exploration(self) -> None:
        """Initialize Phase 1: batch_size exploration."""
        step_size = MAX_CLIENT_BATCH_SIZE // 2  # 150
        min_start = step_size
        self.explore_step = step_size

        initial_max = self.n_instances * MAX_CLIENT_BATCH_SIZE * 3 // 2

        self.explore_values = []
        for size in range(min_start, initial_max + 1, step_size):
            if size <= self.batch_size_max:
                self.explore_values.append(size)

        if not self.explore_values:
            self.explore_values = [min_start]

        self.explore_min_value = self.n_instances * MAX_CLIENT_BATCH_SIZE // 2
        self.explore_index = 0
        self.explore_results = {v: [] for v in self.explore_values}
        self.explore_decline_count = 0
        self.optimal_batch_size = self.explore_values[0]
        self.optimal_max_concurrent = self.n_instances

    def _init_phase2_concurrent_exploration(self) -> None:
        """Initialize Phase 2: max_concurrent exploration."""
        self.explore_phase = 2

        start = max(self.max_concurrent_min, self.n_instances)
        if start % 2 != 0:
            start += 1
        self.explore_values = list(range(start, self.max_concurrent_max + 1, 2))

        if not self.explore_values:
            self.explore_values = [start]

        self.explore_min_value = 0  # No minimum for phase 2
        self.explore_index = 0
        self.explore_results = {v: [] for v in self.explore_values}
        self.explore_decline_count = 0
        self.optimal_max_concurrent = self.explore_values[0]

    def record_measurement(self, throughput: float) -> None:
        """Record a throughput measurement for current exploration value.

        Args:
            throughput: Measured throughput (items/second)
        """
        if not self.exploring or not self.explore_values:
            return

        current_value = self.explore_values[self.explore_index]
        self.explore_results[current_value].append(throughput)

        # Check if we have enough samples
        if len(self.explore_results[current_value]) < self.explore_samples_per_value:
            return

        # Check early stop or advance
        if self._should_stop_exploration():
            self._finalize_exploration()
            return

        self.explore_index += 1

        # Try extending if exhausted
        if self.explore_index >= len(self.explore_values):
            if not self._try_extend_exploration():
                self._finalize_exploration()

    def _should_stop_exploration(self) -> bool:
        """Check if exploration should stop due to performance drop."""
        if len(self.explore_results) < 2:
            return False

        value_throughputs = {}
        for value, throughputs in self.explore_results.items():
            if throughputs:
                value_throughputs[value] = sum(throughputs) / len(throughputs)

        if len(value_throughputs) < 2:
            return False

        tested_values = sorted([v for v in value_throughputs.keys()])
        current_value = tested_values[-1]

        # Phase 1: Never stop before reaching minimum
        if self.explore_phase == 1 and current_value < self.explore_min_value:
            return False

        best_throughput = max(value_throughputs.values())
        current_throughput = value_throughputs[current_value]

        drop_threshold = 0.95
        is_decline = current_throughput < best_throughput * drop_threshold

        if is_decline:
            self.explore_decline_count += 1
        else:
            self.explore_decline_count = 0

        return self.explore_decline_count >= self.explore_decline_max

    def _try_extend_exploration(self) -> bool:
        """Try to extend exploration range."""
        if not self.explore_values:
            return False

        max_tried = max(self.explore_values)

        if self.explore_phase == 1:
            next_value = max_tried + self.explore_step
            max_limit = self.batch_size_max
        else:
            next_value = max_tried + 2
            max_limit = self.max_concurrent_max

        if next_value > max_limit:
            return False

        self.explore_values.append(next_value)
        self.explore_results[next_value] = []
        return True

    def _finalize_exploration(self) -> tuple[int, float]:
        """Finalize current phase and transition to next or finish.

        Returns:
            Tuple of (optimal_value, best_throughput)
        """
        if not self.explore_results:
            return (0, 0.0)

        # Find best value
        best_value = 0
        best_throughput = 0.0

        for value, throughputs in self.explore_results.items():
            if not throughputs:
                continue
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_value = value

        if self.explore_phase == 1:
            # Phase 1 -> Phase 2
            if best_value > 0:
                self.phase1_best_batch = best_value
                self.phase1_best_throughput = best_throughput
                self.optimal_batch_size = best_value
            self._init_phase2_concurrent_exploration()
        else:
            # Phase 2 -> Done
            if best_value > 0:
                self.optimal_max_concurrent = best_value
            self.exploring = False
            self.best_throughput = best_throughput

        return (best_value, best_throughput)

    def get_current_batch_size(self) -> int:
        """Get the batch size to use for next request."""
        if not self.exploring:
            return self.optimal_batch_size

        if self.explore_phase == 1:
            return self.explore_values[self.explore_index]
        else:
            return self.optimal_batch_size

    def get_current_max_concurrent(self) -> int:
        """Get the max_concurrent to use for next request."""
        if not self.exploring:
            return self.optimal_max_concurrent

        if self.explore_phase == 1:
            return self.n_instances
        else:
            return self.explore_values[self.explore_index]


class PerformanceTracker:
    """Combines metrics and exploration for performance tracking.

    Used by benchmark and performance-optimized clients.
    """

    def __init__(
        self,
        n_instances: int,
        enable_exploration: bool = True,
        saved_config: dict | None = None,
    ):
        """Initialize performance tracker.

        Args:
            n_instances: Number of GPU instances
            enable_exploration: Whether to enable exploration
            saved_config: Optional saved configuration to skip exploration
        """
        self.metrics = PerformanceMetrics()
        self.exploration = ExplorationState()

        # Try to load from saved config
        if saved_config and self._load_from_config(saved_config, n_instances):
            enable_exploration = False

        if enable_exploration:
            self.exploration.initialize(n_instances)
        else:
            self.exploration.exploring = False

    def _load_from_config(self, config: dict, n_instances: int) -> bool:
        """Load configuration from saved config."""
        saved_instances = config.get("instances", 0)
        saved_batch = config.get("optimal_batch_size", 0)
        saved_max_concurrent = config.get("optimal_max_concurrent", 0)
        saved_throughput = config.get("throughput", 0.0)

        # Only use if instances match
        if saved_instances != n_instances or saved_batch <= 0:
            return False

        self.exploration.optimal_batch_size = saved_batch
        self.exploration.optimal_max_concurrent = (
            saved_max_concurrent
            if saved_max_concurrent > 0
            else max(6, n_instances * 2)
        )
        self.exploration.best_throughput = saved_throughput
        self.metrics.throughput_ema = saved_throughput

        return True

    def record_request(self, latency: float, n_items: int) -> None:
        """Record a request completion.

        Args:
            latency: Request latency in seconds
            n_items: Number of items processed
        """
        current_throughput = self.metrics.update(latency, n_items)

        if self.exploration.exploring and current_throughput > 0:
            self.exploration.record_measurement(current_throughput)

    def get_batch_size(self) -> int:
        """Get optimal/current batch size."""
        return self.exploration.get_current_batch_size()

    def get_max_concurrent(self) -> int:
        """Get optimal/current max concurrent requests."""
        return self.exploration.get_current_max_concurrent()

    def is_exploring(self) -> bool:
        """Check if still exploring."""
        return self.exploration.exploring

    def get_stats_dict(self, elapsed_time: float = 0.0) -> dict:
        """Get statistics as dictionary.

        Args:
            elapsed_time: Total elapsed time for cumulative throughput calculation

        Returns:
            Dictionary with performance statistics
        """
        return {
            "optimal_batch_size": self.exploration.optimal_batch_size,
            "optimal_max_concurrent": self.exploration.optimal_max_concurrent,
            "exploring": self.exploration.exploring,
            "explore_phase": (
                self.exploration.explore_phase if self.exploration.exploring else 0
            ),
            "throughput_ema": round(self.metrics.throughput_ema, 1),
            "throughput_cumulative": round(
                self.metrics.get_cumulative_throughput(elapsed_time), 1
            ),
            "total_items": self.metrics.total_items,
            "total_requests": self.metrics.total_requests,
        }
