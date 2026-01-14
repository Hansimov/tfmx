"""TEI (Text Embeddings Inference) Multi-Machine Client

This module provides a client for connecting to multiple TEI machines,
with client-side load balancing across machines.
"""

# ANCHOR[id=clients-clis]
CLI_EPILOG = """
Examples:
  export TEI_EPS="http://localhost:28800,http://ai122:28800"
  
  # Note: -E/--endpoints must be placed BEFORE the subcommand
  tei_clients -E $TEI_EPS health
  tei_clients -E $TEI_EPS info
  tei_clients -E $TEI_EPS embed "Hello" "World"
  tei_clients -E $TEI_EPS lsh "Hello"
  tei_clients -E $TEI_EPS lsh -b 2048 "Hello, world"
"""

import argparse
import asyncio
import hashlib
import httpx
import json
import orjson  # Fast JSON serialization
import os
import threading
import time

from dataclasses import dataclass, field
from pathlib import Path
from tclogger import logger
from typing import Union, Optional, Iterable, Iterator

from .tei_client import TEIClient, AsyncTEIClient, InfoResponse, TIMEOUT
from .tei_compose import MAX_CLIENT_BATCH_SIZE

# from .tei_scheduler import IdleFillingScheduler  # Unused


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
            Dict with optimal_batch_size, throughput, instances, updated_at
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
        from datetime import datetime

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
class MachineState:
    """State tracking for a TEI machine with adaptive performance metrics.

    Similar to WorkerState in tei_scheduler.py but for machine-level scheduling.
    Tracks:
    - Busy/idle status for pipeline scheduling
    - Optimal (batch_size, max_concurrent) discovery via two-phase exploration
    - Real-time throughput estimation (EMA)

    Two-Phase Exploration:
    - Phase 1: Explore batch_size with fixed max_concurrent=6
    - Phase 2: Explore max_concurrent with the optimal batch_size from Phase 1
    """

    endpoint: str
    client: TEIClient = field(repr=False)  # Sync client for health checks
    async_client: AsyncTEIClient = field(
        default=None, repr=False
    )  # Async client for pipeline

    # Health status
    healthy: bool = False
    healthy_instances: int = 0
    total_instances: int = 0

    # Config saving (optional)
    _config_saver: Optional["ExplorationConfig"] = field(default=None, repr=False)
    _endpoints_list: list[str] = field(default_factory=list, repr=False)

    # Concurrent request tracking for pipeline scheduling
    # Allow multiple concurrent requests to utilize all GPUs on the machine
    _active_requests: int = 0
    _max_concurrent: int = 6  # Default, will be optimized during exploration

    # Optimal configuration (result of exploration)
    optimal_batch_size: int = MAX_CLIENT_BATCH_SIZE
    optimal_max_concurrent: int = 6
    _batch_size_min: int = 500  # Minimum batch size to try
    _batch_size_max: int = 5000  # Maximum batch size to try
    _max_concurrent_min: int = 2  # Minimum max_concurrent to try
    _max_concurrent_max: int = 20  # Maximum max_concurrent to try

    # Throughput tracking (EMA for real-time estimation)
    _throughput_ema: float = 0.0  # items/second
    _latency_ema: float = 0.0  # seconds per batch
    _ema_alpha: float = 0.3  # EMA smoothing factor

    # Statistics
    _total_items: int = 0
    _total_latency: float = 0.0
    _total_requests: int = 0

    # Two-phase exploration state
    _exploring: bool = True  # Whether we're still exploring
    _explore_phase: int = 1  # 1 = batch_size, 2 = max_concurrent
    _explore_values: list[int] = field(default_factory=list)  # Values to try
    _explore_index: int = 0  # Current index in explore_values
    _explore_results: dict = field(default_factory=dict)  # value -> [throughputs]
    _explore_samples_per_value: int = 3  # Samples to collect per value
    _explore_step: int = 250  # Step size for batch_size exploration
    _explore_n_instances: int = 1  # Number of instances for this machine
    _explore_min_value: int = 0  # Minimum value to explore before allowing early stop
    _explore_decline_count: int = 0  # Consecutive decline count
    _explore_decline_max: int = 3  # Max consecutive declines before stopping
    _best_throughput: float = 0.0  # Best throughput found during exploration
    _phase1_best_batch: int = 0  # Best batch size from phase 1
    _phase1_best_throughput: float = 0.0  # Best throughput from phase 1

    def initialize_exploration(self, n_instances: int) -> None:
        """Initialize three-phase exploration based on number of GPU instances.

        Phase 1: Explore batch_size with max_concurrent = n_instances
        Phase 2: Explore max_concurrent with optimal batch_size from Phase 1
        Phase 3: Refine batch_size with optimal max_concurrent from Phase 2

        Args:
            n_instances: Number of healthy GPU instances for this machine
        """
        self._explore_n_instances = n_instances
        self._explore_phase = 1
        self._exploring = True

        # Initialize Phase 1: batch_size exploration
        self._init_phase1_batch_exploration()

    def _init_phase1_batch_exploration(self) -> None:
        """Initialize Phase 1: batch_size exploration with max_concurrent = n_instances.

        Use exactly n_instances concurrent requests to ensure each GPU gets work,
        which gives a more accurate baseline for batch_size exploration.
        """
        # Use finer granularity: half of MAX_CLIENT_BATCH_SIZE
        step_size = MAX_CLIENT_BATCH_SIZE // 2  # 250
        self._explore_step = step_size

        # Start exploration from step_size, no upper limit yet
        # Initial range: from step_size up to 2x instances * MAX_CLIENT_BATCH_SIZE
        initial_max = self._explore_n_instances * MAX_CLIENT_BATCH_SIZE * 2

        self._explore_values = []
        for size in range(step_size, initial_max + 1, step_size):
            if size <= self._batch_size_max:
                self._explore_values.append(size)

        # Ensure we have at least one value
        if not self._explore_values:
            self._explore_values = [step_size]

        # Minimum value to explore before allowing early stop
        self._explore_min_value = self._explore_n_instances * MAX_CLIENT_BATCH_SIZE

        # Reset exploration state
        self._explore_index = 0
        self._explore_results = {v: [] for v in self._explore_values}
        self._explore_decline_count = 0

        # Start with first value
        # Use n_instances as max_concurrent to ensure each GPU gets exactly one request
        self.optimal_batch_size = self._explore_values[0]
        self._max_concurrent = self._explore_n_instances

    def _init_phase2_concurrent_exploration(self) -> None:
        """Initialize Phase 2: max_concurrent exploration with optimal batch_size from Phase 1."""
        self._explore_phase = 2

        # Use the optimal batch_size from Phase 1
        # max_concurrent range: n_instances to 20, step of 2
        # Start from n_instances since we need at least one concurrent request per GPU
        start = max(self._max_concurrent_min, self._explore_n_instances)
        # Ensure start is even for consistent step
        if start % 2 != 0:
            start += 1
        self._explore_values = list(range(start, self._max_concurrent_max + 1, 2))

        # Minimum value before allowing early stop
        self._explore_min_value = start

        # Reset exploration state
        self._explore_index = 0
        self._explore_results = {v: [] for v in self._explore_values}
        self._explore_decline_count = 0

        # Start with first value (n_instances)
        self._max_concurrent = self._explore_values[0]

    def _init_phase3_batch_refinement(self) -> None:
        """Initialize Phase 3: batch_size refinement with optimal max_concurrent.

        Use the optimal max_concurrent from Phase 2 to refine batch_size.
        This explores around the Phase 1 best batch_size with finer granularity.
        """
        self._explore_phase = 3

        # Get the Phase 1 best batch as center point
        center = self._phase1_best_batch
        step_size = MAX_CLIENT_BATCH_SIZE // 4  # 125 (finer granularity)

        # Explore range: center - 2*step to center + 4*step
        # Bias towards larger batches since more concurrent requests can handle them
        min_val = max(self._batch_size_min, center - 2 * step_size)
        max_val = min(self._batch_size_max, center + 4 * step_size)

        self._explore_values = list(range(min_val, max_val + 1, step_size))

        # Remove the center value if it exists (already tested in Phase 1)
        # Actually keep it for comparison, but it will be quick since we have prior data

        # Minimum value before allowing early stop: center point
        self._explore_min_value = center

        # Reset exploration state
        self._explore_index = 0
        self._explore_results = {v: [] for v in self._explore_values}
        self._explore_decline_count = 0

        # Start with first value, keep optimal_max_concurrent from Phase 2
        self.optimal_batch_size = self._explore_values[0]

    def get_next_batch_size(self) -> int:
        """Get the batch size to use for next request.

        During Phase 1: cycles through different batch sizes
        During Phase 2: returns the optimal batch size from Phase 1
        During Phase 3: cycles through batch sizes for refinement
        After exploration: returns the final optimal batch size
        """
        if not self._exploring:
            return self.optimal_batch_size

        if self._explore_phase == 2:
            # Phase 2: fixed batch_size, exploring max_concurrent
            return self.optimal_batch_size

        # Phase 1 or 3: exploring batch sizes
        if not self._explore_values:
            return self.optimal_batch_size

        return self._explore_values[self._explore_index]

    @property
    def is_idle(self) -> bool:
        """Check if machine can accept more requests."""
        return self._active_requests < self._max_concurrent

    @property
    def busy(self) -> bool:
        """Legacy property for compatibility."""
        return self._active_requests >= self._max_concurrent

    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        return self._active_requests

    @property
    def available_slots(self) -> int:
        """Number of request slots available."""
        return max(0, self._max_concurrent - self._active_requests)

    @property
    def throughput(self) -> float:
        """Get estimated throughput in items/second (EMA)."""
        return self._throughput_ema

    @property
    def latency(self) -> float:
        """Get estimated latency per batch in seconds (EMA)."""
        return self._latency_ema

    @property
    def weight(self) -> int:
        """Weight for load balancing based on healthy instances."""
        return self.healthy_instances if self.healthy else 0

    def get_current_max_concurrent(self) -> int:
        """Get the max_concurrent value to use for current request.

        During Phase 2 exploration: cycles through different values
        Otherwise: returns the current _max_concurrent
        """
        if self._exploring and self._explore_phase == 2 and self._explore_values:
            return self._explore_values[self._explore_index]
        return self._max_concurrent

    def update_max_concurrent_from_exploration(self) -> None:
        """Update _max_concurrent during Phase 2 exploration."""
        if self._exploring and self._explore_phase == 2 and self._explore_values:
            self._max_concurrent = self._explore_values[self._explore_index]

    def mark_busy(self) -> None:
        """Increment active request count."""
        self._active_requests += 1

    def mark_idle(self) -> None:
        """Decrement active request count."""
        self._active_requests = max(0, self._active_requests - 1)

    def _update_ema(self, latency: float, n_items: int) -> float:
        """Update EMA throughput/latency metrics. Returns current throughput."""
        if latency <= 0:
            return 0.0
        current_throughput = n_items / latency
        if self._throughput_ema == 0:
            self._throughput_ema = current_throughput
            self._latency_ema = latency
        else:
            self._throughput_ema = (
                self._ema_alpha * current_throughput
                + (1 - self._ema_alpha) * self._throughput_ema
            )
            self._latency_ema = (
                self._ema_alpha * latency + (1 - self._ema_alpha) * self._latency_ema
            )
        return current_throughput

    def _advance_exploration(self, current_throughput: float) -> None:
        """Advance exploration state if enough samples collected."""
        if not self._exploring or not self._explore_values:
            return

        current_value = self._explore_values[self._explore_index]
        self._explore_results[current_value].append(current_throughput)

        # Not enough samples yet
        if len(self._explore_results[current_value]) < self._explore_samples_per_value:
            return

        # Check early stop or advance to next value
        if self._should_stop_exploration():
            self._finalize_exploration()
            return

        self._explore_index += 1

        # Try extending if exhausted
        if self._explore_index >= len(self._explore_values):
            if not self._try_extend_exploration():
                self._finalize_exploration()
                return

        # Phase 2: sync _max_concurrent with exploration
        if self._exploring and self._explore_phase == 2:
            self._max_concurrent = self._explore_values[self._explore_index]

    def record_success(self, latency: float, n_items: int) -> None:
        """Record a successful request and update metrics."""
        self._total_requests += 1
        self._total_items += n_items
        self._total_latency += latency

        current_throughput = self._update_ema(latency, n_items)
        if current_throughput > 0:
            self._advance_exploration(current_throughput)

    def _should_stop_exploration(self) -> bool:
        """Check if exploration should stop due to performance drop.

        Logic:
        1. Phase 1 (batch_size): Never stop before reaching _explore_min_value
        2. Phase 2 (max_concurrent): No minimum constraint (range is small)
        3. Phase 3 (batch_size refinement): No minimum constraint (fixed range)
        4. Track consecutive declines from best throughput
        5. If performance recovers (within 5% of best), reset decline counter
        6. Stop after _explore_decline_max (3) consecutive declines
        """
        if len(self._explore_results) < 2:
            return False

        # Get average throughputs for tested values
        value_throughputs = {}
        for value, throughputs in self._explore_results.items():
            if throughputs:
                value_throughputs[value] = sum(throughputs) / len(throughputs)

        if len(value_throughputs) < 2:
            return False

        # Get current (last tested) value
        tested_values = sorted([v for v in value_throughputs.keys()])
        current_value = tested_values[-1]

        # Phase-specific minimum checks
        if self._explore_phase == 1:
            # Phase 1: Never stop before reaching minimum batch size
            if current_value < self._explore_min_value:
                return False
        # Phase 2 & 3: No minimum constraint

        # Find best throughput so far
        best_throughput = max(value_throughputs.values())
        current_throughput = value_throughputs[current_value]

        # Check if current is a decline (>5% worse than best)
        drop_threshold = 0.95
        is_decline = current_throughput < best_throughput * drop_threshold

        if is_decline:
            self._explore_decline_count += 1
        else:
            # Performance recovered, reset counter
            self._explore_decline_count = 0

        # Stop after N consecutive declines
        if self._explore_decline_count >= self._explore_decline_max:
            return True

        return False

    def _try_extend_exploration(self) -> bool:
        """Try to extend exploration with more values.

        Phase 1: Extend batch_size range
        Phase 2: Extend max_concurrent range
        Phase 3: No extension (fixed range around Phase 1 best)

        Returns True if a new value was added, False if we've hit the limit.
        """
        if not self._explore_values:
            return False

        # Phase 3 has fixed range, no extension
        if self._explore_phase == 3:
            return False

        # Get the largest value we've tried
        max_tried = max(self._explore_values)

        if self._explore_phase == 1:
            # Phase 1: Extend batch_size
            next_value = max_tried + self._explore_step
            max_limit = self._batch_size_max
        else:
            # Phase 2: Extend max_concurrent (step=2)
            next_value = max_tried + 2
            max_limit = self._max_concurrent_max

        # Check if we've hit the limit
        if next_value > max_limit:
            return False

        # Add the new value
        self._explore_values.append(next_value)
        self._explore_results[next_value] = []

        return True

    def _finalize_exploration(self) -> None:
        """Analyze exploration results and handle phase transitions.

        Phase 1: Find best batch_size (with max_concurrent = n_instances), then start Phase 2
        Phase 2: Find best max_concurrent (with Phase 1 batch_size), then start Phase 3
        Phase 3: Refine batch_size (with Phase 2 max_concurrent), then finish
        """
        if not self._explore_results:
            return

        endpoint_short = self.endpoint.split("/")[-1].split(":")[0]

        # Find best value with best average throughput
        best_value = 0
        best_throughput = 0.0

        for value, throughputs in self._explore_results.items():
            if not throughputs:
                continue
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_value = value

        if self._explore_phase == 1:
            # Phase 1 complete: Save best batch_size and start Phase 2
            if best_value > 0:
                self._phase1_best_batch = best_value
                self._phase1_best_throughput = best_throughput
                self.optimal_batch_size = best_value

            logger.file(
                f"[{endpoint_short}] Phase 1 done: optimal_batch={self.optimal_batch_size}, "
                f"throughput={best_throughput:.0f}/s. Starting Phase 2..."
            )

            # Start Phase 2: Explore max_concurrent
            self._init_phase2_concurrent_exploration()

        elif self._explore_phase == 2:
            # Phase 2 complete: Save best max_concurrent and start Phase 3
            if best_value > 0:
                self.optimal_max_concurrent = best_value
                self._max_concurrent = best_value

            logger.file(
                f"[{endpoint_short}] Phase 2 done: optimal_max_concurrent={self.optimal_max_concurrent}, "
                f"throughput={best_throughput:.0f}/s. Starting Phase 3..."
            )

            # Start Phase 3: Refine batch_size with optimal max_concurrent
            self._init_phase3_batch_refinement()

        else:
            # Phase 3 complete: Save best batch_size and finish
            if best_value > 0:
                self.optimal_batch_size = best_value

            self._exploring = False
            self._best_throughput = best_throughput

            logger.okay(
                f"[{endpoint_short}] Explore done: optimal_batch={self.optimal_batch_size}, "
                f"optimal_max_concurrent={self.optimal_max_concurrent}, "
                f"throughput={best_throughput:.0f}/s"
            )

            # Save config with both optimal values
            self._save_config_if_available()

    def load_from_config(self, config: dict) -> bool:
        """Load optimal batch size and max_concurrent from saved config, skipping exploration.

        Args:
            config: Dict with optimal_batch_size, optimal_max_concurrent, throughput, instances

        Returns:
            True if config was loaded and exploration skipped
        """
        if not config:
            return False

        saved_instances = config.get("instances", 0)
        saved_batch = config.get("optimal_batch_size", 0)
        saved_max_concurrent = config.get("optimal_max_concurrent", 0)
        saved_throughput = config.get("throughput", 0.0)

        # Only use saved config if instances match (GPU configuration unchanged)
        if saved_instances != self.healthy_instances:
            endpoint_short = self.endpoint.split("/")[-1].split(":")[0]
            logger.mesg(
                f"[{endpoint_short}] Config instances mismatch "
                f"(saved={saved_instances}, current={self.healthy_instances}), re-exploring"
            )
            return False

        if saved_batch > 0:
            self.optimal_batch_size = saved_batch
            self._throughput_ema = saved_throughput
            self._exploring = False
            self._best_throughput = saved_throughput

            # Load optimal_max_concurrent if available
            if saved_max_concurrent > 0:
                self.optimal_max_concurrent = saved_max_concurrent
                self._max_concurrent = saved_max_concurrent
            else:
                # Fallback: calculate based on instances
                self._max_concurrent = max(6, self.healthy_instances * 2)

            endpoint_short = self.endpoint.split("/")[-1].split(":")[0]
            logger.okay(
                f"[{endpoint_short}] Loaded config: optimal_batch={saved_batch}, "
                f"max_concurrent={self._max_concurrent}, throughput={saved_throughput:.0f}/s"
            )
            return True

        return False

    def set_config_saver(
        self, config_saver: "ExplorationConfig", endpoints_list: list[str]
    ) -> None:
        """Set the config saver for automatic config persistence.

        Args:
            config_saver: ExplorationConfig instance to use for saving
            endpoints_list: Full list of endpoints for config key generation
        """
        self._config_saver = config_saver
        self._endpoints_list = endpoints_list

    def _save_config_if_available(self) -> None:
        """Save current optimal config to file if saver is available."""
        if (
            self._config_saver
            and self._endpoints_list
            and not self._exploring
            and self._best_throughput > 0
        ):
            self._config_saver.save_machine_config(
                endpoints=self._endpoints_list,
                endpoint=self.endpoint,
                optimal_batch_size=self.optimal_batch_size,
                optimal_max_concurrent=self.optimal_max_concurrent,
                throughput=self._best_throughput,
                instances=self.healthy_instances,
            )

    def to_dict(self, elapsed_time: float = 0.0) -> dict:
        """Convert to dictionary for serialization.

        Args:
            elapsed_time: Total elapsed time in seconds for calculating cumulative throughput.
                         If 0, uses total_latency as fallback.
        """
        # Calculate cumulative throughput (actual throughput during the session)
        if elapsed_time > 0:
            cumulative_throughput = self._total_items / elapsed_time
        elif self._total_latency > 0:
            # Fallback: use total latency (sum of individual request latencies)
            # Note: This underestimates throughput when requests run in parallel
            cumulative_throughput = self._total_items / self._total_latency
        else:
            cumulative_throughput = 0.0

        result = {
            "endpoint": self.endpoint,
            "healthy": self.healthy,
            "healthy_instances": self.healthy_instances,
            "active_requests": self._active_requests,
            "max_concurrent": self._max_concurrent,
            "optimal_batch_size": self.optimal_batch_size,
            "optimal_max_concurrent": self.optimal_max_concurrent,
            "throughput_ema": round(self._throughput_ema, 1),
            "throughput": round(
                cumulative_throughput, 1
            ),  # Actual cumulative throughput
            "latency_ema_ms": round(self._latency_ema * 1000, 1),
            "total_items": self._total_items,
            "total_requests": self._total_requests,
        }

        # Add exploration status if exploring
        if self._exploring:
            result["exploring_phase"] = self._explore_phase
            result["exploring_index"] = (
                f"{self._explore_index + 1}/{len(self._explore_values)}"
            )

        return result


class IteratorBuffer:
    """Thread-safe buffer for pulling items from an iterator on demand.

    Allows multiple async workers to pull batches from a shared iterator
    while maintaining correct ordering of results.
    """

    def __init__(self, iterator: Iterator[str], total_hint: int | None = None):
        """Initialize buffer with an iterator.

        Args:
            iterator: Source iterator to pull items from
            total_hint: Optional hint for total number of items (for progress)
        """
        self._iterator = iterator
        self._lock = threading.Lock()
        self._exhausted = False
        self._next_index = 0  # Next item index to assign
        self._total_hint = total_hint
        self._total_pulled = 0

    def get_batch(self, batch_size: int) -> tuple[int, list[str]]:
        """Pull a batch of items from the iterator.

        Args:
            batch_size: Maximum number of items to pull

        Returns:
            Tuple of (start_index, items_list).
            Returns (start_index, []) when iterator is exhausted.
        """
        with self._lock:
            if self._exhausted:
                return (self._next_index, [])

            items = []
            start_idx = self._next_index

            for _ in range(batch_size):
                try:
                    item = next(self._iterator)
                    items.append(item)
                    self._next_index += 1
                    self._total_pulled += 1
                except StopIteration:
                    self._exhausted = True
                    break

            return (start_idx, items)

    @property
    def exhausted(self) -> bool:
        """Check if iterator is exhausted."""
        with self._lock:
            return self._exhausted

    @property
    def total_pulled(self) -> int:
        """Total number of items pulled from iterator."""
        with self._lock:
            return self._total_pulled

    @property
    def total_hint(self) -> int | None:
        """Hint for total number of items (may be None)."""
        return self._total_hint

    @property
    def remaining_hint(self) -> int | None:
        """Estimate of remaining items (may be None if total_hint not provided)."""
        if self._total_hint is None:
            return None
        with self._lock:
            return max(0, self._total_hint - self._total_pulled)


class MachineScheduler:
    """Pipeline scheduler for distributing work across machines.

    Unlike ratio-based splitting, this scheduler:
    1. Each machine has its own optimal batch size
    2. Machines work independently in a pipeline (no round barriers)
    3. Idle machines immediately get new work
    4. Fast machines naturally process more batches
    5. Allows multiple concurrent requests per machine to keep GPUs fed
    """

    def __init__(self, machines: list[MachineState]):
        self.machines = machines
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # Initially all idle
        self._recent_throughputs: list[float] = []
        self._throughput_window: int = 10

    def get_healthy_machines(self) -> list[MachineState]:
        """Get list of healthy machines."""
        return [m for m in self.machines if m.healthy]

    def get_idle_machine(self) -> Optional[MachineState]:
        """Get a machine with available slots, preferring ones with more capacity."""
        idle = [m for m in self.machines if m.healthy and m.is_idle]
        if not idle:
            self._idle_event.clear()
            return None
        idle.sort(key=lambda m: (m.available_slots, m.throughput), reverse=True)
        return idle[0]

    def signal_idle(self) -> None:
        """Signal that a machine has become idle."""
        self._idle_event.set()

    def calc_tail_batch_size(
        self, base_size: int, remaining: int | None, total_capacity: int
    ) -> int:
        """Calculate optimized batch size for tail distribution.

        When remaining items are fewer than total capacity, distribute evenly
        across idle slots to maximize parallelism.
        """
        if remaining is None or remaining >= total_capacity:
            return base_size
        total_idle = sum(m.available_slots for m in self.machines if m.healthy)
        if total_idle <= 0:
            return base_size
        ideal = (remaining + total_idle - 1) // total_idle
        return max(100, min(base_size, ideal))

    def record_throughput(self, throughput: float) -> None:
        """Record a throughput measurement."""
        self._recent_throughputs.append(throughput)
        if len(self._recent_throughputs) > self._throughput_window:
            self._recent_throughputs.pop(0)

    @property
    def recent_throughput(self) -> float:
        """Get recent average throughput."""
        if not self._recent_throughputs:
            return 0.0
        return sum(self._recent_throughputs) / len(self._recent_throughputs)

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "machines": [m.to_dict() for m in self.machines],
            "recent_throughput": round(self.recent_throughput, 1),
            "healthy_count": sum(1 for m in self.machines if m.healthy),
            "total_max_concurrent": sum(
                m._max_concurrent for m in self.machines if m.healthy
            ),
        }


@dataclass
class ClientsHealthResponse:
    """Health response for the multi-machine clients."""

    status: str
    healthy_machines: int
    total_machines: int
    healthy_instances: int
    total_instances: int

    @classmethod
    def from_machines(cls, machines: list[MachineState]) -> "ClientsHealthResponse":
        healthy_machines = sum(1 for m in machines if m.healthy)
        healthy_instances = sum(m.healthy_instances for m in machines)
        total_instances = sum(m.total_instances for m in machines)
        return cls(
            status="healthy" if healthy_machines > 0 else "unhealthy",
            healthy_machines=healthy_machines,
            total_machines=len(machines),
            healthy_instances=healthy_instances,
            total_instances=total_instances,
        )


class TEIClients:
    """Multi-machine TEI client with client-side load balancing.

    Connects to multiple tei_machine endpoints and distributes requests
    across them for maximum throughput.

    Example:
        clients = TEIClients([
            "http://machine1:28800",
            "http://machine2:28800",
        ])
        embs = clients.embed(["Hello", "World"])
        clients.close()

    With context manager:
        with TEIClients(["http://m1:28800", "http://m2:28800"]) as clients:
            embs = clients.embed(["Hello", "World"])
    """

    def __init__(
        self,
        endpoints: list[str],
        timeout: float = TIMEOUT,
        verbose: bool = False,
        skip_exploration: bool = False,
    ):
        """Initialize multi-machine TEI client.

        Args:
            endpoints: List of tei_machine endpoint URLs
                      (e.g., ["http://machine1:28800", "http://machine2:28800"])
            timeout: Request timeout in seconds (default: 60.0)
            verbose: Enable verbose logging
            skip_exploration: If True, try to load saved exploration config
                             instead of re-exploring batch sizes
        """
        self.endpoints = [ep.rstrip("/") for ep in endpoints]
        self.timeout = timeout
        self.verbose = verbose
        self.skip_exploration = skip_exploration

        # Exploration config manager
        self._exploration_config = ExplorationConfig()

        # Create underlying clients for each endpoint (sync for health checks)
        self.clients: list[TEIClient] = [
            TEIClient(endpoint=ep, timeout=timeout, verbose=verbose)
            for ep in self.endpoints
        ]

        # Create async clients for each endpoint (for high-throughput pipeline)
        self.async_clients: list[AsyncTEIClient] = [
            AsyncTEIClient(endpoint=ep, timeout=timeout, verbose=verbose)
            for ep in self.endpoints
        ]

        # Machine states for pipeline scheduling
        self.machines: list[MachineState] = [
            MachineState(endpoint=ep, client=sync_client, async_client=async_client)
            for ep, sync_client, async_client in zip(
                self.endpoints, self.clients, self.async_clients
            )
        ]

        # Set config saver for all machines (for automatic config persistence)
        for machine in self.machines:
            machine.set_config_saver(self._exploration_config, self.endpoints)

        # Pipeline scheduler for machine-level distribution
        self.machine_scheduler = MachineScheduler(self.machines)

        # Round-robin index (for small input fallback)
        self._rr_index = 0

    def close(self) -> None:
        """Close all HTTP clients."""
        for client in self.clients:
            client.close()

    async def aclose(self) -> None:
        """Close all async HTTP clients."""
        for async_client in self.async_clients:
            await async_client.close()

    def __enter__(self) -> "TEIClients":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def refresh_health(self) -> ClientsHealthResponse:
        """Refresh health status of all machines.

        Returns:
            ClientsHealthResponse with aggregated health info.
        """
        for machine in self.machines:
            self._refresh_machine_health(machine)
        return ClientsHealthResponse.from_machines(self.machines)

    def _refresh_machine_health(self, machine: MachineState) -> None:
        """Refresh health for a single machine and initialize exploration if needed."""
        try:
            health = machine.client.health()
            was_healthy = machine.healthy
            machine.healthy = health.status == "healthy" or health.healthy > 0
            machine.healthy_instances = health.healthy
            machine.total_instances = health.total

            # Initialize on first healthy check or when exploration not started
            needs_init = machine.healthy and (
                not was_healthy or not machine._explore_values
            )
            if not needs_init:
                return

            # Try loading saved config first
            if self.skip_exploration:
                saved = self._exploration_config.get_machine_config(
                    self.endpoints, machine.endpoint
                )
                if saved and machine.load_from_config(saved):
                    return

            # Fall back to exploration
            machine.initialize_exploration(machine.healthy_instances)
            short_name = machine.endpoint.split("/")[-1].split(":")[0]
            logger.mesg(
                f"[{short_name}] {machine.healthy_instances} GPUs, "
                f"exploring Phase {machine._explore_phase}"
            )
        except Exception:
            machine.healthy = False
            machine.healthy_instances = 0

    def health(self) -> ClientsHealthResponse:
        """Check health status of all machines.

        Returns:
            ClientsHealthResponse with aggregated health info.
        """
        return self.refresh_health()

    def save_exploration_config(self) -> None:
        """Save exploration results for all machines to config file.

        Called after exploration is complete to persist optimal batch sizes.
        """
        for machine in self.machines:
            if machine.healthy and not machine._exploring:
                self._exploration_config.save_machine_config(
                    endpoints=self.endpoints,
                    endpoint=machine.endpoint,
                    optimal_batch_size=machine.optimal_batch_size,
                    throughput=machine._best_throughput,
                    instances=machine.healthy_instances,
                )

    def clear_exploration_config(self) -> None:
        """Clear saved exploration config for current endpoints."""
        self._exploration_config.clear(self.endpoints)
        logger.mesg("Cleared saved exploration config")

    def _ensure_healthy(self) -> list[MachineState]:
        """Ensure healthy machines are available, refreshing if needed."""
        healthy = self.machine_scheduler.get_healthy_machines()
        if not healthy:
            self.refresh_health()
            healthy = self.machine_scheduler.get_healthy_machines()
        if not healthy:
            raise ValueError("No healthy machines available")
        return healthy

    def _run_pipeline(
        self,
        inputs: list[str] | Iterator[str],
        healthy: list[MachineState],
        request_fn,  # async fn(machine, chunk) -> results
        action_name: str = "pipeline",
        total_hint: int | None = None,
    ) -> list:
        """Generic pipeline for distributing batch requests across machines.

        Args:
            inputs: List or iterator of input texts
            healthy: List of healthy machines to use
            request_fn: Async function (machine, chunk) -> results
            action_name: Name for logging
            total_hint: Optional total count hint for iterator inputs

        Returns:
            Combined results in input order
        """
        # Determine if inputs is a list or iterator
        if isinstance(inputs, list):
            buffer = IteratorBuffer(iter(inputs), len(inputs))
        else:
            buffer = IteratorBuffer(inputs, total_hint)

        results_map: dict[int, list] = {}
        pending_tasks: set[asyncio.Task] = set()
        errors: list[tuple[str, Exception]] = []
        batch_count = 0
        scheduler = self.machine_scheduler

        # Per-machine tracking for progress stats
        machine_stats: dict[str, dict] = {
            m.endpoint: {"items": 0, "host": m.endpoint.split("//")[-1].split(":")[0]}
            for m in healthy
        }

        # Calculate total capacity for tail optimization
        total_capacity = sum(m.optimal_batch_size * m._max_concurrent for m in healthy)

        async def process_batch(
            machine: MachineState, chunk: list[str], start_idx: int
        ):
            """Execute request and return (machine, start_idx, results, latency, error)."""
            task_start = time.perf_counter()
            try:
                results = await request_fn(machine, chunk)
                return (
                    machine,
                    start_idx,
                    results,
                    time.perf_counter() - task_start,
                    None,
                )
            except Exception as e:
                return (machine, start_idx, None, time.perf_counter() - task_start, e)

        def get_batch_size(machine: MachineState) -> int:
            """Get batch size with tail optimization."""
            base = machine.get_next_batch_size()
            return scheduler.calc_tail_batch_size(
                base, buffer.remaining_hint, total_capacity
            )

        def dispatch_batch(machine: MachineState) -> asyncio.Task | None:
            """Try to dispatch a batch to machine. Returns task or None."""
            nonlocal batch_count
            batch_size = get_batch_size(machine)
            start_idx, chunk = buffer.get_batch(batch_size)
            if not chunk:
                return None
            batch_count += 1
            machine.mark_busy()
            task = asyncio.create_task(process_batch(machine, chunk, start_idx))
            task._start_idx = start_idx  # type: ignore
            return task

        def handle_result(machine, start_idx, results, latency, error):
            """Process a completed task result."""
            if error is None and results is not None:
                results_map[start_idx] = results
                machine.record_success(latency, len(results))
                if latency > 0:
                    scheduler.record_throughput(len(results) / latency)
                # Track per-machine stats
                stats = machine_stats[machine.endpoint]
                stats["items"] += len(results)
            else:
                machine.healthy = False
                errors.append((machine.endpoint, error or Exception("Unknown error")))

        async def run():
            nonlocal pending_tasks
            session_start = time.perf_counter()
            total_processed = 0
            last_log_pct = 0

            while not buffer.exhausted or pending_tasks:
                # Dispatch work to all idle machines
                while not buffer.exhausted:
                    machine = scheduler.get_idle_machine()
                    if not machine:
                        break
                    task = dispatch_batch(machine)
                    if task:
                        pending_tasks.add(task)
                    else:
                        break

                if pending_tasks:
                    await asyncio.sleep(0)  # Let tasks start

                if not pending_tasks:
                    break

                # Wait for completion
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # First: mark idle and prepare new batches (minimize dispatch gap)
                new_tasks = []
                completed = []
                for task in done:
                    machine, start_idx, results, latency, error = task.result()
                    completed.append((machine, start_idx, results, latency, error))
                    machine.mark_idle()
                    scheduler.signal_idle()
                    # Try to dispatch new work immediately
                    if not buffer.exhausted and machine.is_idle:
                        new_task = dispatch_batch(machine)
                        if new_task:
                            new_tasks.append(new_task)

                pending_tasks.update(new_tasks)
                if new_tasks:
                    await asyncio.sleep(0)

                # Then: process results
                for machine, start_idx, results, latency, error in completed:
                    handle_result(machine, start_idx, results, latency, error)
                    if results:
                        total_processed += len(results)

                # Progress logging for large batches
                if buffer.total_hint and buffer.total_hint >= 10000:
                    pct = int(total_processed / buffer.total_hint * 100)
                    if pct >= last_log_pct + 10:
                        elapsed = time.perf_counter() - session_start
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        # Build per-machine stats string: host:throughput
                        ep_stats = ", ".join(
                            (
                                f"{s['host']}:{s['items']/elapsed:.0f}/s"
                                if elapsed > 0
                                else f"{s['host']}:0/s"
                            )
                            for s in machine_stats.values()
                        )
                        logger.mesg(
                            f"  [{pct:3d}%] {total_processed:,}/{buffer.total_hint:,} | "
                            f"{rate:,.0f}/s | {ep_stats}"
                        )
                        last_log_pct = pct

            return time.perf_counter() - session_start

        total_time = asyncio.run(run())

        if not results_map:
            raise ValueError(f"All requests failed: {errors}")

        # Combine in order
        combined = []
        for idx in sorted(results_map.keys()):
            combined.extend(results_map[idx])

        if self.verbose:
            throughput = len(combined) / total_time if total_time > 0 else 0
            logger.okay(
                f"[Pipeline] {action_name}: {len(combined)} items, {batch_count} batches, "
                f"{total_time:.2f}s, {throughput:.0f}/s"
            )

        return combined

    def embed(
        self,
        inputs: Union[str, list[str]],
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for input texts using multiple machines.

        Args:
            inputs: Single text or list of texts to embed.
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            List of embedding vectors (list of floats).

        Raises:
            ValueError: When no healthy machines available or all requests fail
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            return []

        healthy = self._ensure_healthy()

        # Small inputs: single machine, round-robin
        if len(inputs) <= 10:
            machine = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return machine.client.embed(inputs, normalize=normalize, truncate=truncate)

        # Single machine: direct call
        if len(healthy) == 1:
            return healthy[0].client.embed(
                inputs, normalize=normalize, truncate=truncate
            )

        # Multiple machines: pipeline
        return self._run_pipeline(
            inputs=inputs,
            healthy=healthy,
            request_fn=lambda m, chunk: m.async_client.embed(
                chunk, normalize=normalize, truncate=truncate
            ),
            action_name="embed",
        )

    def lsh(
        self,
        inputs: Union[str, list[str]],
        bitn: int = 2048,
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[str]:
        """Generate LSH hash hex strings for input texts using multiple machines.

        Args:
            inputs: Single text or list of texts.
            bitn: Number of LSH hash bits (default: 2048, range: 64-8192)
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            List of hex strings representing LSH hashes.

        Raises:
            ValueError: When no healthy machines available or all requests fail
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            return []

        healthy = self._ensure_healthy()

        # Small inputs: single machine, round-robin
        if len(inputs) <= 10:
            machine = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return machine.client.lsh(
                inputs, bitn=bitn, normalize=normalize, truncate=truncate
            )

        # Single machine: direct call
        if len(healthy) == 1:
            return healthy[0].client.lsh(
                inputs, bitn=bitn, normalize=normalize, truncate=truncate
            )

        # Multiple machines: pipeline
        return self._run_pipeline(
            inputs=inputs,
            healthy=healthy,
            request_fn=lambda m, chunk: m.async_client.lsh(
                chunk, bitn=bitn, normalize=normalize, truncate=truncate
            ),
            action_name="lsh",
        )

    def lsh_iter(
        self,
        inputs: Iterable[str],
        total_hint: int | None = None,
        bitn: int = 2048,
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[str]:
        """Generate LSH hashes for an iterable of texts using pipeline scheduling.

        Optimized for large datasets where you don't want to materialize
        the entire input list in memory.

        Args:
            inputs: Iterable of texts (can be generator, iterator, or list)
            total_hint: Optional hint for total number of items (for progress logging)
            bitn: Number of LSH hash bits (default: 2048)
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            List of hex strings representing LSH hashes, in input order.
        """
        healthy = self._ensure_healthy()
        return self._run_pipeline(
            inputs=iter(inputs),
            healthy=healthy,
            request_fn=lambda m, chunk: m.async_client.lsh(
                chunk, bitn=bitn, normalize=normalize, truncate=truncate
            ),
            action_name="lsh",
            total_hint=total_hint,
        )

    def get_scheduler_stats(self) -> dict:
        """Get scheduler statistics."""
        return self.machine_scheduler.get_stats()

    def get_machine_stats(self, elapsed_time: float = 0.0) -> list[dict]:
        """Get statistics for all machines.

        Args:
            elapsed_time: Total elapsed time for calculating cumulative throughput.
        """
        return [m.to_dict(elapsed_time=elapsed_time) for m in self.machines]

    def info(self) -> list[InfoResponse]:
        """Get info from all machines.

        Returns:
            List of InfoResponse from each machine.
        """
        responses = []
        for machine in self.machines:
            try:
                responses.append(machine.client.info())
            except Exception:
                pass
        return responses


class TEIClientsArgParser:
    """Argument parser for TEI Clients CLI."""

    def __init__(self):
        # Create main parser with common arguments at root level
        self.parser = argparse.ArgumentParser(
            description="TEI Clients - Connect to multiple TEI machines",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )

        # Add common arguments to main parser
        self._add_common_arguments(self.parser)

        # Setup subcommands (they won't have these common arguments repeated)
        self._setup_subcommands()
        self.args = self.parser.parse_args()

    def _add_common_arguments(self, parser):
        """Add common arguments to a parser.

        This method centralizes the definition of arguments that can appear
        either before or after the subcommand.
        """
        parser.add_argument(
            "-E",
            "--endpoints",
            type=str,
            required=False,
            help="Comma-separated list of tei_machine endpoints",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

    def _setup_subcommands(self):
        """Setup subcommands."""
        # Action subcommands
        subparsers = self.parser.add_subparsers(dest="action", help="Action to perform")

        # health
        subparsers.add_parser(
            "health",
            help="Check health of all machines",
        )

        # info
        subparsers.add_parser(
            "info",
            help="Get info from all machines",
        )

        # embed
        embed_parser = subparsers.add_parser(
            "embed",
            help="Generate embeddings",
        )
        embed_parser.add_argument(
            "texts",
            nargs="+",
            help="Texts to embed",
        )

        # lsh
        lsh_parser = subparsers.add_parser(
            "lsh",
            help="Generate LSH hashes",
        )
        lsh_parser.add_argument(
            "texts",
            nargs="+",
            help="Texts to hash",
        )
        lsh_parser.add_argument(
            "-b",
            "--bitn",
            type=int,
            default=2048,
            help="Number of LSH bits (default: 2048)",
        )


class TEIClientsCLI:
    """CLI interface for TEI Clients operations."""

    def __init__(self, clients: TEIClients):
        """Initialize CLI with TEI clients.

        Args:
            clients: TEIClients instance to use for operations
        """
        self.clients = clients

    def run_health(self) -> None:
        """Run health check and display results."""
        machines = self.clients.machines
        if not machines:
            logger.warn("× No machine info available")
            return

        for i, machine in enumerate(machines):
            logger.note(f"[Machine {i+1}] {machine.endpoint}")
            machine.client.log_machine_health()

    def run_info(self) -> None:
        """Get and display info from all machines."""
        machines = self.clients.machines
        if not machines:
            logger.warn("× No machine info available")
            return

        for i, machine in enumerate(machines):
            logger.okay(f"[Machine {i+1}] {machine.endpoint}")
            machine.client.log_machine_info()
            print()

    def run_embed(self, texts: list[str]) -> None:
        """Generate and display embeddings.

        Args:
            texts: List of texts to embed
        """
        if not texts:
            logger.warn("× No input texts provided")
            return

        embs = self.clients.embed(texts)
        print(json.dumps(embs, indent=2))

    def run_lsh(self, texts: list[str], bitn: int = 2048) -> None:
        """Generate and display LSH hashes.

        Args:
            texts: List of texts to hash
            bitn: Number of LSH bits
        """
        if not texts:
            logger.warn("× No input texts provided")
            return

        hashes = self.clients.lsh(texts, bitn=bitn)
        for text, hash_str in zip(texts, hashes):
            text_preview = text[:40] + "..." if len(text) > 40 else text
            hash_preview = hash_str[:32] + "..." if len(hash_str) > 32 else hash_str
            logger.mesg(f"'{text_preview}'")
            logger.file(f"  → {hash_preview}")


def main():
    """Main entry point for CLI."""
    arg_parser = TEIClientsArgParser()
    args = arg_parser.args

    if args.action is None:
        arg_parser.parser.print_help()
        return

    # Validate endpoints argument
    if not args.endpoints:
        logger.warn("× Error: -E/--endpoints is required")
        arg_parser.parser.print_help()
        return

    endpoints = [ep.strip() for ep in args.endpoints.split(",")]
    clients = TEIClients(
        endpoints=endpoints,
        verbose=args.verbose,
    )

    try:
        cli = TEIClientsCLI(clients)
        if args.action == "health":
            cli.run_health()
        elif args.action == "info":
            cli.run_info()
        elif args.action == "embed":
            cli.run_embed(args.texts)
        elif args.action == "lsh":
            cli.run_lsh(args.texts, args.bitn)
    except httpx.ConnectError as e:
        logger.warn(f"× Connection failed: {e}")
        logger.hint(f"  Check if all TEI machines are running")
    except Exception as e:
        logger.warn(f"× Error: {e}")
    finally:
        clients.close()


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_clients.py#clients-clis
