"""TEI Clients Core - Shared Infrastructure for Multi-Machine Clients

This module contains the core components shared between production (TEIClients)
and stats/exploration versions (TEIClientsWithStats):

- MachineState: Machine health and request tracking
- MachineScheduler: Pipeline scheduling logic
- IteratorBuffer: Thread-safe iterator buffering
- ClientsHealthResponse: Health status aggregation
- _TEIClientsPipeline: Core pipeline implementation (composition component)
- _TEIClientsBase: Abstract base class with shared method implementations

Design: Uses composition pattern for pipeline + inheritance for shared methods.
"""

import asyncio
import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Iterator, Callable, Any, Union, Iterable

from .tei_client import TEIClient, AsyncTEIClient, InfoResponse
from .tei_compose import MAX_CLIENT_BATCH_SIZE
from .tei_performance import ExplorationConfig


@dataclass
class MachineState:
    """State tracking for a TEI machine.

    Tracks health status and concurrent requests for pipeline scheduling.
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

    # Concurrent request tracking
    _active_requests: int = 0
    _max_concurrent: int = 6

    # Batch size configuration
    batch_size: int = MAX_CLIENT_BATCH_SIZE

    # Capacity configuration (for proportional scheduling)
    # Base capacity from config (throughput per second)
    _config_throughput: float = 0.0
    _config_instances: int = 0

    @property
    def is_idle(self) -> bool:
        """Check if machine can accept more requests."""
        return self._active_requests < self._max_concurrent

    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        return self._active_requests

    @property
    def available_slots(self) -> int:
        """Number of request slots available."""
        return max(0, self._max_concurrent - self._active_requests)

    @property
    def weight(self) -> int:
        """Weight for load balancing based on healthy instances."""
        return self.healthy_instances if self.healthy else 0

    @property
    def capacity(self) -> float:
        """Effective capacity for proportional scheduling.

        Returns throughput scaled by healthy/config instances ratio.
        If no config, falls back to batch_size as capacity proxy.
        """
        if self._config_throughput > 0 and self._config_instances > 0:
            scale = self.healthy_instances / self._config_instances
            return self._config_throughput * scale
        # Fallback: use batch_size * max_concurrent as capacity proxy
        return float(self.batch_size * self._max_concurrent) if self.healthy else 0.0

    def mark_busy(self) -> None:
        """Increment active request count."""
        self._active_requests += 1

    def mark_idle(self) -> None:
        """Decrement active request count."""
        self._active_requests = max(0, self._active_requests - 1)


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

    Features:
    1. Each machine has its own optimal batch size
    2. Machines work independently in a pipeline (no round barriers)
    3. Idle machines immediately get new work
    4. Fast machines naturally process more batches
    5. Allows multiple concurrent requests per machine to keep GPUs fed
    6. Proportional batch allocation based on capacity
    """

    def __init__(self, machines: list[MachineState]):
        self.machines = machines
        self._idle_event = asyncio.Event()
        self._idle_event.set()  # Initially all idle

    def get_healthy_machines(self) -> list[MachineState]:
        """Get list of healthy machines."""
        return [m for m in self.machines if m.healthy]

    def get_idle_machine(self) -> Optional[MachineState]:
        """Get a machine with available slots, preferring ones with more capacity."""
        idle = [m for m in self.machines if m.healthy and m.is_idle]
        if not idle:
            self._idle_event.clear()
            return None
        idle.sort(key=lambda m: m.available_slots, reverse=True)
        return idle[0]

    def signal_idle(self) -> None:
        """Signal that a machine has become idle."""
        self._idle_event.set()

    def get_total_capacity(self, healthy: list[MachineState] = None) -> float:
        """Get total capacity of all healthy machines."""
        if healthy is None:
            healthy = self.get_healthy_machines()
        return sum(m.capacity for m in healthy)

    def calc_proportional_batch_size(
        self, machine: MachineState, total_remaining: int, healthy: list[MachineState]
    ) -> int:
        """Calculate batch size for a machine based on capacity proportion.

        Uses a target cycle time (e.g., 0.5s) to determine batch size, rather than
        giving each machine its full share of remaining work. This ensures:
        1. Multiple dispatch cycles, keeping all machines continuously busy
        2. Fast machines get more batches, not just larger batches
        3. Better overlap between machine processing and client-side work

        Args:
            machine: The machine to calculate batch size for
            total_remaining: Total number of items remaining to process
            healthy: List of healthy machines

        Returns:
            Batch size to assign to this machine
        """
        # Target cycle time in seconds - how much work to dispatch per round
        # Smaller = more frequent dispatches = better overlap, but more overhead
        # 0.5s is a good balance: allows 2-3 dispatch cycles during a typical batch
        TARGET_CYCLE_TIME = 0.5

        # Calculate batch size based on throughput * cycle time
        if machine._config_throughput > 0:
            # Use configured throughput to estimate one cycle's worth of data
            # Scale by actual instances vs config instances
            if machine._config_instances > 0:
                scale = machine.healthy_instances / machine._config_instances
            else:
                scale = 1.0
            effective_throughput = machine._config_throughput * scale
            cycle_batch_size = int(effective_throughput * TARGET_CYCLE_TIME)
        else:
            # Fallback: use batch_size / max_concurrent as one cycle's worth
            cycle_batch_size = machine.batch_size // max(machine._max_concurrent, 1)

        # Ensure minimum batch size for efficiency
        MIN_BATCH_SIZE = 200
        cycle_batch_size = max(cycle_batch_size, MIN_BATCH_SIZE)

        # Clamp to machine's batch_size limit and remaining
        batch_size = min(cycle_batch_size, machine.batch_size, total_remaining)

        return max(1, batch_size)

    def calc_tail_batch_size(
        self, base_size: int, remaining: int | None, total_capacity: int
    ) -> int:
        """Calculate optimized batch size for tail distribution.

        Strategy: Keep batch_size stable to avoid scheduling chaos.
        Modern async scheduler handles tail efficiently without manual optimization.
        """
        # Simply return base_size - no tail optimization
        # This avoids creating many small batches at the end which hurts throughput
        return base_size


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


class _TEIClientsPipeline:
    """Core pipeline implementation for distributing requests across machines.

    This class encapsulates the async pipeline logic and can be composed into
    both production and stats-enabled client classes. Uses callbacks for
    extensibility (logging, stats collection, etc.).

    Composition pattern: This is an internal component that handles the
    complex async orchestration. Parent classes use it via composition and
    can optionally provide callbacks for logging/stats.
    """

    def __init__(
        self,
        machine_scheduler: MachineScheduler,
        on_progress: Optional[Callable[[int, int, float, dict], None]] = None,
        on_complete: Optional[Callable[[int, int, float], None]] = None,
    ):
        """Initialize pipeline.

        Args:
            machine_scheduler: Scheduler managing machine states
            on_progress: Optional callback(processed, total, elapsed, machine_stats)
                        called periodically during execution (for logging)
            on_complete: Optional callback(total_items, batch_count, total_time)
                        called after pipeline completes (for logging)
        """
        self.scheduler = machine_scheduler
        self.on_progress = on_progress
        self.on_complete = on_complete

    def run_pipeline(
        self,
        inputs: list[str] | Iterator[str],
        healthy: list[MachineState],
        request_fn: Callable[[MachineState, list[str]], Any],
        action_name: str = "pipeline",
        total_hint: int | None = None,
        close_clients: bool = True,
    ) -> list:
        """Execute async pipeline distributing work across machines.

        Args:
            inputs: List or iterator of input texts
            healthy: List of healthy machines to use
            request_fn: Async function (machine, chunk) -> results
            action_name: Name for logging (e.g., "embed", "lsh")
            total_hint: Optional total count hint for iterator inputs
            close_clients: Whether to close async clients after completion (default: True)

        Returns:
            Combined results in input order
        """
        # Reset all async clients for use in new event loop
        # (Previous asyncio.run() may have closed the old event loop)
        for m in healthy:
            if m.async_client:
                m.async_client.reset()

        total_time = asyncio.run(
            self._run_pipeline_async(
                inputs=inputs,
                healthy=healthy,
                request_fn=request_fn,
                action_name=action_name,
                total_hint=total_hint,
                close_clients=close_clients,
            )
        )
        return total_time

    async def run_pipeline_async(
        self,
        inputs: list[str] | Iterator[str],
        healthy: list[MachineState],
        request_fn: Callable[[MachineState, list[str]], Any],
        action_name: str = "pipeline",
        total_hint: int | None = None,
        close_clients: bool = False,
    ) -> list:
        """Async version of run_pipeline for use in existing event loops.

        This method is designed to be called from within an async context,
        allowing connection reuse across multiple calls without the overhead
        of creating new event loops or reconnecting HTTP clients.

        Args:
            inputs: List or iterator of input texts
            healthy: List of healthy machines to use
            request_fn: Async function (machine, chunk) -> results
            action_name: Name for logging (e.g., "embed", "lsh")
            total_hint: Optional total count hint for iterator inputs
            close_clients: Whether to close async clients after completion (default: False)

        Returns:
            Combined results in input order
        """
        return await self._run_pipeline_async(
            inputs=inputs,
            healthy=healthy,
            request_fn=request_fn,
            action_name=action_name,
            total_hint=total_hint,
            close_clients=close_clients,
        )

    async def _run_pipeline_async(
        self,
        inputs: list[str] | Iterator[str],
        healthy: list[MachineState],
        request_fn: Callable[[MachineState, list[str]], Any],
        action_name: str = "pipeline",
        total_hint: int | None = None,
        close_clients: bool = True,
    ) -> list:
        """Internal async implementation of the pipeline.

        Args:
            inputs: List or iterator of input texts
            healthy: List of healthy machines to use
            request_fn: Async function (machine, chunk) -> results
            action_name: Name for logging (e.g., "embed", "lsh")
            total_hint: Optional total count hint for iterator inputs
            close_clients: Whether to close async clients after completion

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

        # Per-machine tracking for progress stats
        machine_stats: dict[str, dict] = {
            m.endpoint: {"items": 0, "host": m.endpoint.split("//")[-1].split(":")[0]}
            for m in healthy
        }

        # Calculate total capacity for tail optimization
        total_capacity = self.scheduler.get_total_capacity(healthy)

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
            """Get batch size using proportional allocation based on capacity.

            Uses capacity-based proportional sizing when remaining items are known,
            otherwise falls back to machine's configured batch_size.
            """
            remaining = buffer.remaining_hint

            # If we have remaining hint, use proportional allocation
            if remaining is not None and remaining > 0:
                return self.scheduler.calc_proportional_batch_size(
                    machine, remaining, healthy
                )

            # Fallback to configured batch_size
            return machine.batch_size

        def calc_proportional_shares(
            idle_machines: list[MachineState], remaining: int
        ) -> dict[str, int]:
            """Calculate batch sizes for all idle machines using cycle-based allocation.

            Uses target cycle time to determine batch size per machine, rather than
            splitting remaining work proportionally. This ensures multiple dispatch
            cycles, keeping all machines continuously busy.
            """
            if not idle_machines or remaining <= 0:
                return {}

            # Target cycle time - same as in calc_proportional_batch_size
            TARGET_CYCLE_TIME = 0.5
            MIN_BATCH_SIZE = 200

            shares = {}
            allocated = 0

            for m in idle_machines:
                if remaining - allocated <= 0:
                    shares[m.endpoint] = 0
                    continue

                # Calculate cycle-based batch size
                if m._config_throughput > 0:
                    if m._config_instances > 0:
                        scale = m.healthy_instances / m._config_instances
                    else:
                        scale = 1.0
                    effective_throughput = m._config_throughput * scale
                    cycle_batch_size = int(effective_throughput * TARGET_CYCLE_TIME)
                else:
                    cycle_batch_size = m.batch_size // max(m._max_concurrent, 1)

                cycle_batch_size = max(cycle_batch_size, MIN_BATCH_SIZE)

                # Clamp to machine's batch_size limit and remaining
                share = min(cycle_batch_size, m.batch_size, remaining - allocated)
                share = max(1, share) if share > 0 else 0
                shares[m.endpoint] = share
                allocated += share

            return shares

        def dispatch_batch(
            machine: MachineState, batch_size: int = None
        ) -> asyncio.Task | None:
            """Try to dispatch a batch to machine. Returns task or None.

            Args:
                machine: Machine to dispatch to
                batch_size: Optional pre-calculated batch size (for proportional allocation)
            """
            nonlocal batch_count
            if batch_size is None:
                batch_size = get_batch_size(machine)
            start_idx, chunk = buffer.get_batch(batch_size)
            if not chunk:
                return None
            batch_count += 1
            machine.mark_busy()
            task = asyncio.create_task(process_batch(machine, chunk, start_idx))
            return task

        def dispatch_to_idle_machines_proportionally() -> list[asyncio.Task]:
            """Dispatch batches to all idle machines using proportional allocation.

            Calculates shares atomically before taking any batches.
            """
            idle = [m for m in healthy if m.is_idle]
            if not idle:
                return []

            remaining = buffer.remaining_hint
            if remaining is None:
                # Fallback: dispatch one by one with default batch sizes
                tasks = []
                for machine in idle:
                    task = dispatch_batch(machine)
                    if task:
                        tasks.append(task)
                    else:
                        break
                return tasks

            # Calculate proportional shares for all idle machines at once
            shares = calc_proportional_shares(idle, remaining)

            tasks = []
            for machine in idle:
                batch_size = shares.get(machine.endpoint, 0)
                if batch_size > 0:
                    task = dispatch_batch(machine, batch_size)
                    if task:
                        tasks.append(task)
            return tasks

        def handle_result(machine, start_idx, results, latency, error):
            """Process a completed task result."""
            if error is None and results is not None:
                results_map[start_idx] = results
                # Track per-machine stats
                stats = machine_stats[machine.endpoint]
                stats["items"] += len(results)
            else:
                machine.healthy = False
                errors.append((machine.endpoint, error or Exception("Unknown error")))

        session_start = time.perf_counter()
        total_processed = 0
        last_log_time = 0.0  # Last progress log time

        while not buffer.exhausted or pending_tasks:
            # Dispatch work to all idle machines using proportional allocation
            if not buffer.exhausted:
                new_dispatch_tasks = dispatch_to_idle_machines_proportionally()
                pending_tasks.update(new_dispatch_tasks)

            if pending_tasks:
                await asyncio.sleep(0)  # Let tasks start

            if not pending_tasks:
                break

            # Wait for completion
            done, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # First: mark idle and collect completed tasks
            completed = []
            for task in done:
                machine, start_idx, results, latency, error = task.result()
                completed.append((machine, start_idx, results, latency, error))
                machine.mark_idle()
                self.scheduler.signal_idle()

            # Dispatch new work to all newly idle machines proportionally
            if not buffer.exhausted:
                new_tasks = dispatch_to_idle_machines_proportionally()
                pending_tasks.update(new_tasks)
                if new_tasks:
                    await asyncio.sleep(0)

            # Then: process results
            for machine, start_idx, results, latency, error in completed:
                handle_result(machine, start_idx, results, latency, error)
                if results:
                    total_processed += len(results)

            # Progress callback: trigger every 5 seconds
            if self.on_progress and buffer.total_hint and buffer.total_hint >= 1000:
                elapsed = time.perf_counter() - session_start
                if elapsed - last_log_time >= 5.0:  # Every 5 seconds
                    self.on_progress(
                        total_processed, buffer.total_hint, elapsed, machine_stats
                    )
                    last_log_time = elapsed

        # Close all async clients if requested (typically when using asyncio.run)
        if close_clients:
            for m in healthy:
                if m.async_client and m.async_client._client:
                    await m.async_client.close()

        total_time = time.perf_counter() - session_start

        if not results_map:
            raise ValueError(f"All requests failed: {errors}")

        # Combine in order
        combined = []
        for idx in sorted(results_map.keys()):
            combined.extend(results_map[idx])

        # Completion callback
        if self.on_complete:
            self.on_complete(len(combined), batch_count, total_time)

        return combined


class _TEIClientsBase(ABC):
    """Abstract base class for multi-machine TEI clients.

    Provides all shared method implementations. Subclasses only need to:
    1. Implement __init__ (with their specific parameters)
    2. Initialize self._pipeline with appropriate callbacks
    3. Optionally override _load_config() for verbose logging

    This eliminates ~380 lines of duplicated code between production
    and stats-enabled versions.
    """

    def __init__(self, endpoints: list[str]):
        """Base initialization - subclasses should call this via super().

        Args:
            endpoints: List of tei_machine endpoint URLs
        """
        self.endpoints = [ep.rstrip("/") for ep in endpoints]

        # Create underlying clients for each endpoint
        # Note: verbose parameter must be set by subclass before calling super()
        verbose = getattr(self, "_verbose", False)
        self.clients: list[TEIClient] = [
            TEIClient(endpoint=ep, verbose=verbose) for ep in self.endpoints
        ]

        # Create async clients for pipeline
        self.async_clients: list[AsyncTEIClient] = [
            AsyncTEIClient(endpoint=ep, verbose=verbose) for ep in self.endpoints
        ]

        # Machine states for pipeline scheduling
        self.machines: list[MachineState] = [
            MachineState(endpoint=ep, client=sync_client, async_client=async_client)
            for ep, sync_client, async_client in zip(
                self.endpoints, self.clients, self.async_clients
            )
        ]

        # Refresh health to get actual healthy instances before loading config
        for machine in self.machines:
            self._refresh_machine_health(machine)

        # Load optimal batch sizes from config (with instance-based scaling)
        self._load_config()

        # Pipeline scheduler
        self.machine_scheduler = MachineScheduler(self.machines)

        # Pipeline executor - subclass must set this
        self._pipeline: Optional[_TEIClientsPipeline] = None

        # Round-robin index for small batches
        self._rr_index = 0

    def _apply_machine_config(
        self, machine: MachineState, saved: dict
    ) -> tuple[bool, int, int, int]:
        """Apply saved config to machine with instance-based scaling.

        Args:
            machine: Machine state to configure
            saved: Saved config dict from ExplorationConfig

        Returns:
            Tuple of (scaled, config_instances, optimal_batch_size, optimal_max_concurrent)
        """
        config_instances = saved.get("instances", 0)
        actual_instances = machine.healthy_instances
        optimal_batch_size = saved.get("optimal_batch_size", machine.batch_size)
        optimal_max_concurrent = saved.get(
            "optimal_max_concurrent", machine._max_concurrent
        )
        config_throughput = saved.get("throughput", 0.0)

        # Store config values for capacity calculation
        machine._config_throughput = config_throughput
        machine._config_instances = config_instances

        # Scale both batch_size and max_concurrent if instances differ
        scaled = False
        if (
            config_instances > 0
            and actual_instances > 0
            and actual_instances != config_instances
        ):
            scale_ratio = actual_instances / config_instances
            machine.batch_size = int(optimal_batch_size * scale_ratio)
            machine._max_concurrent = int(optimal_max_concurrent * scale_ratio)
            scaled = True
        else:
            machine.batch_size = optimal_batch_size
            machine._max_concurrent = optimal_max_concurrent

        return scaled, config_instances, optimal_batch_size, optimal_max_concurrent

    def _load_config(self) -> None:
        """Load optimal configurations from saved config file.

        Scales batch_size proportionally if actual healthy instances
        differs from config's saved instances. max_concurrent stays unchanged.

        Subclasses can override to add verbose logging.
        """
        config = ExplorationConfig()
        for machine in self.machines:
            saved = config.get_machine_config(self.endpoints, machine.endpoint)
            if saved:
                self._apply_machine_config(machine, saved)

    def close(self) -> None:
        """Close all HTTP clients."""
        for client in self.clients:
            client.close()

    async def aclose(self) -> None:
        """Close all async HTTP clients."""
        for async_client in self.async_clients:
            await async_client.close()

    def __enter__(self):
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
        """Refresh health for a single machine."""
        try:
            health = machine.client.health()
            machine.healthy = health.status == "healthy" or health.healthy > 0
            machine.healthy_instances = health.healthy
            machine.total_instances = health.total
        except Exception:
            machine.healthy = False
            machine.healthy_instances = 0

    def health(self) -> ClientsHealthResponse:
        """Check health status of all machines.

        Returns:
            ClientsHealthResponse with aggregated health info.
        """
        return self.refresh_health()

    def _ensure_healthy(self) -> list[MachineState]:
        """Ensure healthy machines are available, refreshing if needed."""
        healthy = self.machine_scheduler.get_healthy_machines()
        if not healthy:
            self.refresh_health()
            healthy = self.machine_scheduler.get_healthy_machines()
        if not healthy:
            raise ValueError("No healthy machines available")
        return healthy

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
        return self._pipeline.run_pipeline(
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
        return self._pipeline.run_pipeline(
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
        return self._pipeline.run_pipeline(
            inputs=iter(inputs),
            healthy=healthy,
            request_fn=lambda m, chunk: m.async_client.lsh(
                chunk, bitn=bitn, normalize=normalize, truncate=truncate
            ),
            action_name="lsh",
            total_hint=total_hint,
        )

    async def lsh_iter_async(
        self,
        inputs: Iterable[str],
        total_hint: int | None = None,
        bitn: int = 2048,
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[str]:
        """Async version of lsh_iter for use in existing event loops.

        This method allows connection reuse across multiple calls, avoiding
        the overhead of creating new event loops and reconnecting HTTP clients.
        Ideal for high-throughput pipeline scenarios.

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
        return await self._pipeline.run_pipeline_async(
            inputs=iter(inputs),
            healthy=healthy,
            request_fn=lambda m, chunk: m.async_client.lsh(
                chunk, bitn=bitn, normalize=normalize, truncate=truncate
            ),
            action_name="lsh",
            total_hint=total_hint,
            close_clients=False,  # Keep connections alive for reuse
        )

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
