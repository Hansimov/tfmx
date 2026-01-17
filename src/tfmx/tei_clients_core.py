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
    ) -> list:
        """Execute async pipeline distributing work across machines.

        Args:
            inputs: List or iterator of input texts
            healthy: List of healthy machines to use
            request_fn: Async function (machine, chunk) -> results
            action_name: Name for logging (e.g., "embed", "lsh")
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

        # Per-machine tracking for progress stats
        machine_stats: dict[str, dict] = {
            m.endpoint: {"items": 0, "host": m.endpoint.split("//")[-1].split(":")[0]}
            for m in healthy
        }

        # Calculate total capacity for tail optimization
        total_capacity = sum(m.batch_size * m._max_concurrent for m in healthy)

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
            base = machine.batch_size
            return self.scheduler.calc_tail_batch_size(
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
            last_log_time = 0.0  # Last progress log time

            while not buffer.exhausted or pending_tasks:
                # Dispatch work to all idle machines
                while not buffer.exhausted:
                    machine = self.scheduler.get_idle_machine()
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
                    self.scheduler.signal_idle()
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

                # Progress callback: trigger every 5 seconds
                if self.on_progress and buffer.total_hint and buffer.total_hint >= 1000:
                    elapsed = time.perf_counter() - session_start
                    if elapsed - last_log_time >= 5.0:  # Every 5 seconds
                        self.on_progress(
                            total_processed, buffer.total_hint, elapsed, machine_stats
                        )
                        last_log_time = elapsed

            return time.perf_counter() - session_start

        total_time = asyncio.run(run())

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

        # Load optimal batch sizes from config
        self._load_config()

        # Pipeline scheduler
        self.machine_scheduler = MachineScheduler(self.machines)

        # Pipeline executor - subclass must set this
        self._pipeline: Optional[_TEIClientsPipeline] = None

        # Round-robin index for small batches
        self._rr_index = 0

    def _load_config(self) -> None:
        """Load optimal configurations from saved config file.

        Subclasses can override to add verbose logging.
        """
        from .tei_performance import ExplorationConfig

        config = ExplorationConfig()
        for machine in self.machines:
            saved = config.get_machine_config(self.endpoints, machine.endpoint)
            if saved:
                machine.batch_size = saved.get("optimal_batch_size", machine.batch_size)
                machine._max_concurrent = saved.get(
                    "optimal_max_concurrent", machine._max_concurrent
                )

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
