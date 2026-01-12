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
import httpx
import json
import threading
import time

from dataclasses import dataclass, field
from tclogger import logger
from typing import Union, Optional, Iterable, Iterator

from .tei_client import TEIClient, InfoResponse, TIMEOUT
from .tei_compose import MAX_CLIENT_BATCH_SIZE
from .tei_scheduler import IdleFillingScheduler, distribute_with_scheduler


@dataclass
class MachineState:
    """State tracking for a TEI machine with adaptive performance metrics.

    Similar to WorkerState in tei_scheduler.py but for machine-level scheduling.
    Tracks:
    - Busy/idle status for pipeline scheduling
    - Optimal batch size discovery via exploration
    - Real-time throughput estimation (EMA)
    """

    endpoint: str
    client: TEIClient = field(repr=False)

    # Health status
    healthy: bool = False
    healthy_instances: int = 0
    total_instances: int = 0

    # Busy/idle state for pipeline scheduling
    busy: bool = False

    # Optimal batch size tracking
    # Start with default, then adapt based on performance
    optimal_batch_size: int = MAX_CLIENT_BATCH_SIZE
    _batch_size_min: int = 500  # Minimum batch size to try
    _batch_size_max: int = 5000  # Maximum batch size to try

    # Throughput tracking (EMA for real-time estimation)
    _throughput_ema: float = 0.0  # items/second
    _latency_ema: float = 0.0  # seconds per batch
    _ema_alpha: float = 0.3  # EMA smoothing factor

    # Statistics
    _total_items: int = 0
    _total_latency: float = 0.0
    _total_requests: int = 0

    # Batch size exploration state
    _exploring: bool = True  # Whether we're still exploring batch sizes
    _explore_sizes: list[int] = field(default_factory=list)  # Sizes to try
    _explore_index: int = 0  # Current index in explore_sizes
    _explore_results: dict = field(default_factory=dict)  # size -> [throughputs]
    _explore_samples_per_size: int = 3  # Samples to collect per size before moving on

    def initialize_exploration(self, n_instances: int) -> None:
        """Initialize batch size exploration based on number of GPU instances.

        Args:
            n_instances: Number of healthy GPU instances for this machine
        """
        # Calculate exploration sizes based on instances
        # Each GPU can handle ~500 items efficiently, so scale by instance count
        base_size = 500
        max_size = min(n_instances * 700, self._batch_size_max)

        # Generate exploration sizes: base, 2x, 3x, ..., up to max
        self._explore_sizes = []
        for multiplier in range(1, 10):  # Up to 9x base size
            size = base_size * multiplier
            if size <= max_size:
                self._explore_sizes.append(size)
            else:
                break

        # Ensure we have at least one size
        if not self._explore_sizes:
            self._explore_sizes = [base_size]

        # Start with first size for exploration
        self.optimal_batch_size = self._explore_sizes[0]

        # Reset exploration state
        self._explore_index = 0
        self._explore_results = {size: [] for size in self._explore_sizes}
        self._exploring = True

    def get_next_batch_size(self) -> int:
        """Get the batch size to use for next request.

        During exploration, cycles through different sizes to test.
        After exploration, returns the optimal batch size.
        """
        if not self._exploring:
            return self.optimal_batch_size

        # If exploration not initialized, return default
        if not self._explore_sizes:
            return self.optimal_batch_size

        # Return current exploration size
        return self._explore_sizes[self._explore_index]

    @property
    def is_idle(self) -> bool:
        return not self.busy

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

    def mark_busy(self) -> None:
        self.busy = True

    def mark_idle(self) -> None:
        self.busy = False

    def record_success(self, latency: float, n_items: int) -> None:
        """Record a successful request and update metrics."""
        self._total_requests += 1
        self._total_items += n_items
        self._total_latency += latency

        # Update EMA throughput
        if latency > 0:
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
                    self._ema_alpha * latency
                    + (1 - self._ema_alpha) * self._latency_ema
                )

            # Record for batch size exploration
            if self._exploring and self._explore_sizes:
                current_size = self._explore_sizes[self._explore_index]
                self._explore_results[current_size].append(current_throughput)

                # Check if we have enough samples for this size
                if (
                    len(self._explore_results[current_size])
                    >= self._explore_samples_per_size
                ):
                    # Move to next size
                    self._explore_index += 1

                    # Check if exploration complete
                    if self._explore_index >= len(self._explore_sizes):
                        self._finalize_exploration()

        self.mark_idle()

    def _finalize_exploration(self) -> None:
        """Analyze exploration results and select optimal batch size."""
        if not self._explore_results:
            return

        # Find batch size with best average throughput
        best_batch = self.optimal_batch_size
        best_throughput = 0.0

        for batch_size, throughputs in self._explore_results.items():
            if not throughputs:
                continue
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_batch = batch_size

        if best_batch > 0:
            self.optimal_batch_size = best_batch

        self._exploring = False

        # Log exploration result
        endpoint_short = self.endpoint.split("/")[-1].split(":")[0]
        logger.success(
            f"[{endpoint_short}] Exploration done: optimal_batch={self.optimal_batch_size}, "
            f"throughput={best_throughput:.0f}/s"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "endpoint": self.endpoint,
            "healthy": self.healthy,
            "healthy_instances": self.healthy_instances,
            "busy": self.busy,
            "optimal_batch_size": self.optimal_batch_size,
            "throughput_ema": round(self._throughput_ema, 1),
            "latency_ema_ms": round(self._latency_ema * 1000, 1),
            "total_items": self._total_items,
            "total_requests": self._total_requests,
        }


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


class MachineScheduler:
    """Pipeline scheduler for distributing work across machines.

    Unlike ratio-based splitting, this scheduler:
    1. Each machine has its own optimal batch size
    2. Machines work independently in a pipeline (no round barriers)
    3. Idle machines immediately get new work
    4. Fast machines naturally process more batches

    This is similar to IdleFillingScheduler but at the machine level.
    """

    def __init__(self, machines: list[MachineState]):
        self.machines = machines

        # Event to signal when a machine becomes idle
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # Initially all idle

        # Real-time throughput tracking
        self._recent_throughputs: list[float] = []  # Last N throughput measurements
        self._throughput_window: int = 10  # Window size for recent throughput

    def get_idle_machines(self) -> list[MachineState]:
        """Get list of idle healthy machines."""
        return [m for m in self.machines if m.healthy and m.is_idle]

    def get_idle_machine(self) -> Optional[MachineState]:
        """Get a single idle machine, preferring higher throughput ones."""
        idle = self.get_idle_machines()
        if not idle:
            self._idle_event.clear()
            return None
        # Sort by throughput (highest first)
        idle.sort(key=lambda m: m.throughput, reverse=True)
        return idle[0]

    def signal_idle(self) -> None:
        """Signal that a machine has become idle."""
        self._idle_event.set()

    async def wait_for_idle(self, timeout: float = 60.0) -> bool:
        """Wait for a machine to become idle."""
        try:
            await asyncio.wait_for(self._idle_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def record_throughput(self, throughput: float) -> None:
        """Record a throughput measurement for real-time tracking."""
        self._recent_throughputs.append(throughput)
        if len(self._recent_throughputs) > self._throughput_window:
            self._recent_throughputs.pop(0)

    @property
    def recent_throughput(self) -> float:
        """Get recent average throughput (EMA-style)."""
        if not self._recent_throughputs:
            return 0.0
        return sum(self._recent_throughputs) / len(self._recent_throughputs)

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "machines": [m.to_dict() for m in self.machines],
            "recent_throughput": round(self.recent_throughput, 1),
            "healthy_count": sum(1 for m in self.machines if m.healthy),
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
    ):
        """Initialize multi-machine TEI client.

        Args:
            endpoints: List of tei_machine endpoint URLs
                      (e.g., ["http://machine1:28800", "http://machine2:28800"])
            timeout: Request timeout in seconds (default: 60.0)
            verbose: Enable verbose logging
        """
        self.endpoints = [ep.rstrip("/") for ep in endpoints]
        self.timeout = timeout
        self.verbose = verbose

        # Create underlying clients for each endpoint
        self.clients: list[TEIClient] = [
            TEIClient(endpoint=ep, timeout=timeout, verbose=verbose)
            for ep in self.endpoints
        ]

        # Machine states for pipeline scheduling
        self.machines: list[MachineState] = [
            MachineState(endpoint=ep, client=client)
            for ep, client in zip(self.endpoints, self.clients)
        ]

        # Pipeline scheduler for machine-level distribution
        self.machine_scheduler = MachineScheduler(self.machines)

        # Round-robin index (for simple fallback)
        self._rr_index = 0

        # Idle-filling scheduler (kept for single-machine GPU-level scheduling)
        self.scheduler = IdleFillingScheduler(
            workers=self.clients,
            get_worker_id=lambda c: c.endpoint,
            max_batch_size=MAX_CLIENT_BATCH_SIZE,
        )

    def close(self) -> None:
        """Close all HTTP clients."""
        for client in self.clients:
            client.close()

    def __enter__(self) -> "TEIClients":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _log_fail(self, action: str, error: Exception) -> None:
        """Log error message."""
        if self.verbose:
            logger.warn(f"× TEIClients {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        """Log success message."""
        if self.verbose:
            logger.okay(f"✓ TEIClients {action}: {message}")

    def _get_healthy_machines(self) -> list[MachineState]:
        """Get list of healthy machines."""
        return [m for m in self.machines if m.healthy]

    def refresh_health(self) -> ClientsHealthResponse:
        """Refresh health status of all machines.

        Also initializes batch size exploration for newly healthy machines.

        Returns:
            ClientsHealthResponse with aggregated health info.
        """
        for machine in self.machines:
            try:
                health = machine.client.health()
                was_healthy = machine.healthy
                machine.healthy = health.status == "healthy" or health.healthy > 0
                machine.healthy_instances = health.healthy
                machine.total_instances = health.total

                # Initialize exploration if machine became healthy or instances changed
                if machine.healthy and (
                    not was_healthy or machine._explore_sizes == []
                ):
                    machine.initialize_exploration(machine.healthy_instances)
                    endpoint_short = machine.endpoint.split("/")[-1].split(":")[0]
                    logger.mesg(
                        f"[{endpoint_short}] {machine.healthy_instances} GPUs, "
                        f"exploring sizes: {machine._explore_sizes}"
                    )
            except Exception:
                machine.healthy = False
                machine.healthy_instances = 0

        return ClientsHealthResponse.from_machines(self.machines)

    def health(self) -> ClientsHealthResponse:
        """Check health status of all machines.

        Returns:
            ClientsHealthResponse with aggregated health info.
        """
        return self.refresh_health()

    def embed(
        self,
        inputs: Union[str, list[str]],
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for input texts using multiple machines.

        Uses pipeline scheduling: each machine gets batches of its optimal size,
        and idle machines immediately get new work.

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

        # Refresh health if needed
        healthy = self._get_healthy_machines()
        if not healthy:
            self.refresh_health()
            healthy = self._get_healthy_machines()

        if not healthy:
            raise ValueError("No healthy machines available")

        # For small inputs, use simple single-machine path
        if len(inputs) <= 10:
            machine = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return machine.client.embed(inputs, normalize=normalize, truncate=truncate)

        # Single machine: send directly
        if len(healthy) == 1:
            return healthy[0].client.embed(
                inputs, normalize=normalize, truncate=truncate
            )

        # Multiple machines: use pipeline scheduling
        return self._embed_with_pipeline(inputs, healthy, normalize, truncate)

    def _embed_with_pipeline(
        self,
        inputs: list[str],
        healthy: list[MachineState],
        normalize: bool,
        truncate: bool,
    ) -> list[list[float]]:
        """Distribute embed requests using pipeline scheduling.

        Each machine processes batches of its optimal size independently.
        No round barriers - idle machines immediately get new work.
        """
        n_inputs = len(inputs)

        # Results storage (indexed by start position for ordering)
        results_map: dict[int, list[list[float]]] = {}
        input_index = 0  # Current position in inputs
        pending_tasks: set[asyncio.Task] = set()
        errors: list[tuple[str, Exception]] = []

        async def process_batch(
            machine: MachineState,
            chunk: list[str],
            start_idx: int,
        ) -> tuple[
            MachineState, int, list[list[float]] | None, float, Exception | None
        ]:
            """Process a batch on a machine."""
            task_start = time.perf_counter()
            try:
                results = await asyncio.to_thread(
                    machine.client.embed, chunk, normalize=normalize, truncate=truncate
                )
                latency = time.perf_counter() - task_start
                return (machine, start_idx, results, latency, None)
            except Exception as e:
                latency = time.perf_counter() - task_start
                return (machine, start_idx, None, latency, e)

        async def run_pipeline():
            nonlocal input_index, results_map, errors, pending_tasks

            session_start = time.perf_counter()

            while input_index < n_inputs or pending_tasks:
                # Try to assign work to idle machines
                while input_index < n_inputs:
                    machine = self.machine_scheduler.get_idle_machine()
                    if not machine:
                        break

                    # Get batch size (uses exploration logic)
                    batch_size = min(
                        machine.get_next_batch_size(), n_inputs - input_index
                    )
                    if batch_size <= 0:
                        break

                    # Extract chunk
                    start_idx = input_index
                    chunk = inputs[input_index : input_index + batch_size]
                    input_index += batch_size

                    # Mark machine as busy and create task
                    machine.mark_busy()
                    task = asyncio.create_task(process_batch(machine, chunk, start_idx))
                    task._start_idx = start_idx  # type: ignore
                    pending_tasks.add(task)

                # If no pending tasks, we're done
                if not pending_tasks:
                    break

                # Wait for at least one task to complete
                done, pending_tasks_new = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                pending_tasks = pending_tasks_new

                # Process completed tasks
                for task in done:
                    machine, start_idx, results, latency, error = task.result()

                    if error is None and results is not None:
                        results_map[start_idx] = results
                        n_items = len(results)
                        machine.record_success(latency, n_items)

                        # Record throughput for monitoring
                        if latency > 0:
                            throughput = n_items / latency
                            self.machine_scheduler.record_throughput(throughput)
                    else:
                        machine.healthy = False
                        machine.mark_idle()
                        errors.append(
                            (machine.endpoint, error or Exception("Unknown error"))
                        )

                    # Signal that machine is idle
                    self.machine_scheduler.signal_idle()

            return time.perf_counter() - session_start

        # Run the pipeline
        total_time = asyncio.run(run_pipeline())

        if not results_map:
            raise ValueError(f"All requests failed: {errors}")

        # Combine results in order
        combined = []
        for start_idx in sorted(results_map.keys()):
            combined.extend(results_map[start_idx])

        # Log stats
        if self.verbose:
            throughput = len(combined) / total_time if total_time > 0 else 0
            logger.success(
                f"[Pipeline] embed: {len(combined)} items in {total_time*1000:.0f}ms "
                f"({throughput:.0f}/s), recent_avg={self.machine_scheduler.recent_throughput:.0f}/s"
            )

        return combined

    def lsh(
        self,
        inputs: Union[str, list[str]],
        bitn: int = 2048,
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[str]:
        """Generate LSH hash hex strings for input texts using multiple machines.

        Uses pipeline scheduling: each machine gets batches of its optimal size,
        and idle machines immediately get new work.

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

        # Refresh health if needed
        healthy = self._get_healthy_machines()
        if not healthy:
            self.refresh_health()
            healthy = self._get_healthy_machines()

        if not healthy:
            raise ValueError("No healthy machines available")

        # For small inputs, use simple single-machine path
        if len(inputs) <= 10:
            machine = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return machine.client.lsh(
                inputs, bitn=bitn, normalize=normalize, truncate=truncate
            )

        # Single machine: send directly
        if len(healthy) == 1:
            return healthy[0].client.lsh(
                inputs, bitn=bitn, normalize=normalize, truncate=truncate
            )

        # Multiple machines: use pipeline scheduling
        return self._lsh_with_pipeline(
            iter(inputs), len(inputs), healthy, bitn, normalize, truncate
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

        This method is optimized for large datasets where you don't want to
        materialize the entire input list in memory. It pulls batches from
        the iterator on-demand based on machine availability.

        Args:
            inputs: Iterable of texts (can be generator, iterator, or list)
            total_hint: Optional hint for total number of items (for progress logging)
            bitn: Number of LSH hash bits (default: 2048)
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            List of hex strings representing LSH hashes, in input order.
        """
        # Convert to iterator if not already
        iterator = iter(inputs)

        # Refresh health if needed
        healthy = self._get_healthy_machines()
        if not healthy:
            self.refresh_health()
            healthy = self._get_healthy_machines()

        if not healthy:
            raise ValueError("No healthy machines available")

        return self._lsh_with_pipeline(
            iterator, total_hint, healthy, bitn, normalize, truncate
        )

    def _lsh_with_pipeline(
        self,
        inputs: Iterator[str],
        total_hint: int | None,
        healthy: list[MachineState],
        bitn: int,
        normalize: bool,
        truncate: bool,
    ) -> list[str]:
        """Distribute LSH requests using pipeline scheduling with iterator input.

        Each machine processes batches of its optimal size independently.
        No round barriers - idle machines immediately get new work.
        Batches are pulled from the iterator on-demand.
        """
        # Create buffer for thread-safe iterator access
        buffer = IteratorBuffer(inputs, total_hint)

        # Results storage (indexed by start position for ordering)
        results_map: dict[int, list[str]] = {}
        pending_tasks: set[asyncio.Task] = set()
        errors: list[tuple[str, Exception]] = []

        # Stats for logging
        batch_count = 0

        async def process_batch(
            machine: MachineState,
            chunk: list[str],
            start_idx: int,
            batch_id: int,
        ) -> tuple[MachineState, int, list[str] | None, float, int, Exception | None]:
            """Process a batch on a machine."""
            task_start = time.perf_counter()
            try:
                results = await asyncio.to_thread(
                    machine.client.lsh,
                    chunk,
                    bitn=bitn,
                    normalize=normalize,
                    truncate=truncate,
                )
                latency = time.perf_counter() - task_start
                return (machine, start_idx, results, latency, batch_id, None)
            except Exception as e:
                latency = time.perf_counter() - task_start
                return (machine, start_idx, None, latency, batch_id, e)

        async def run_pipeline():
            nonlocal results_map, errors, batch_count, pending_tasks

            session_start = time.perf_counter()
            total_processed = 0
            last_log_pct = 0
            log_interval_pct = 10  # Log every 10%

            while not buffer.exhausted or pending_tasks:
                # Try to assign work to idle machines
                while not buffer.exhausted:
                    machine = self.machine_scheduler.get_idle_machine()
                    if not machine:
                        break

                    # Get batch size (uses exploration logic)
                    batch_size = machine.get_next_batch_size()

                    # Pull a batch from the iterator
                    start_idx, chunk = buffer.get_batch(batch_size)
                    if not chunk:
                        break  # Iterator exhausted

                    batch_count += 1

                    # Mark machine as busy and create task
                    machine.mark_busy()
                    task = asyncio.create_task(
                        process_batch(machine, chunk, start_idx, batch_count)
                    )
                    task._start_idx = start_idx  # type: ignore
                    pending_tasks.add(task)

                # If no pending tasks, we're done
                if not pending_tasks:
                    break

                # Wait for at least one task to complete
                done, pending_tasks_new = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                pending_tasks = pending_tasks_new

                # Process completed tasks
                for task in done:
                    machine, start_idx, results, latency, batch_id, error = (
                        task.result()
                    )

                    if error is None and results is not None:
                        results_map[start_idx] = results
                        n_items = len(results)
                        total_processed += n_items
                        machine.record_success(latency, n_items)

                        # Record throughput
                        if latency > 0:
                            throughput = n_items / latency
                            self.machine_scheduler.record_throughput(throughput)

                        # Progress logging (only when total is known and large)
                        if total_hint and total_hint >= 10000:
                            current_pct = int(total_processed / total_hint * 100)
                            if current_pct >= last_log_pct + log_interval_pct:
                                elapsed = time.perf_counter() - session_start
                                rate = total_processed / elapsed if elapsed > 0 else 0
                                machine_stats = ", ".join(
                                    f"{m.endpoint.split('/')[-1].split(':')[0]}:{m.throughput:.0f}/s"
                                    for m in healthy
                                )
                                logger.mesg(
                                    f"  [{current_pct:3d}%] {total_processed:,}/{total_hint:,} | "
                                    f"{rate:,.0f}/s | {machine_stats}"
                                )
                                last_log_pct = current_pct
                    else:
                        machine.healthy = False
                        machine.mark_idle()
                        errors.append(
                            (machine.endpoint, error or Exception("Unknown error"))
                        )

                    # Signal that machine is idle
                    self.machine_scheduler.signal_idle()

            return time.perf_counter() - session_start

        # Run the pipeline
        total_time = asyncio.run(run_pipeline())

        if not results_map:
            raise ValueError(f"All requests failed: {errors}")

        # Combine results in order
        combined = []
        for start_idx in sorted(results_map.keys()):
            combined.extend(results_map[start_idx])

        # Only log in verbose mode
        if self.verbose:
            throughput = len(combined) / total_time if total_time > 0 else 0
            machine_stats = ", ".join(
                f"{m.endpoint.split('/')[-1]}:{m._total_items}" for m in healthy
            )
            n_total = total_hint or len(combined)
            logger.mesg(
                f"  [Pipeline] {len(combined):,}/{n_total:,} items, {batch_count} batches, "
                f"{total_time:.2f}s, {throughput:.0f}/s | {machine_stats}"
            )

        return combined

    def get_scheduler_stats(self) -> dict:
        """Get scheduler statistics."""
        return self.machine_scheduler.get_stats()

    def get_machine_stats(self) -> list[dict]:
        """Get statistics for all machines."""
        return [m.to_dict() for m in self.machines]

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
