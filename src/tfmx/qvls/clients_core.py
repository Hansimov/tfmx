"""QVL Clients Core - Shared Infrastructure for Multi-Machine QVL Clients

Core components for distributing chat completion requests across
multiple vLLM machines:

- MachineState: Machine health and request tracking
- MachineScheduler: Request distribution logic
- ClientsHealthResponse: Health status aggregation
- _QVLClientsPipeline: Core pipeline implementation
- _QVLClientsBase: Abstract base class with shared method implementations
"""

import asyncio
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Union

from .client import QVLClient, AsyncQVLClient, ChatResponse
from .compose import MAX_CONCURRENT_REQUESTS
from .performance import ExplorationConfig


@dataclass
class MachineState:
    """State tracking for a QVL machine."""

    endpoint: str
    client: QVLClient = field(repr=False)
    async_client: AsyncQVLClient = field(default=None, repr=False)

    # Health status
    healthy: bool = False
    healthy_instances: int = 0
    total_instances: int = 0

    # Concurrent request tracking
    _active_requests: int = 0
    _max_concurrent: int = MAX_CONCURRENT_REQUESTS

    # Health recovery tracking
    _consecutive_failures: int = 0
    _last_failure_time: float = 0.0
    _recovery_in_progress: bool = False

    # Config for capacity calculation
    _config_throughput: float = 0.0
    _config_instances: int = 0

    @property
    def is_idle(self) -> bool:
        return self._active_requests < self._max_concurrent

    @property
    def active_requests(self) -> int:
        return self._active_requests

    @property
    def available_slots(self) -> int:
        return max(0, self._max_concurrent - self._active_requests)

    @property
    def weight(self) -> int:
        return self.healthy_instances if self.healthy else 0

    @property
    def capacity(self) -> float:
        if self._config_throughput > 0 and self._config_instances > 0:
            scale = self.healthy_instances / self._config_instances
            return self._config_throughput * scale
        return float(self._max_concurrent) if self.healthy else 0.0

    def mark_busy(self) -> None:
        self._active_requests += 1

    def mark_idle(self) -> None:
        self._active_requests = max(0, self._active_requests - 1)

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_failure_time = time.time()

    def record_success(self) -> None:
        self._consecutive_failures = 0


class HealthRecoveryManager:
    """Manages background health recovery for unhealthy machines."""

    RECOVERY_CHECK_INTERVAL = 10.0
    MAX_FAILURES_BEFORE_BACKOFF = 3
    BACKOFF_MULTIPLIER = 2.0

    def __init__(
        self,
        machines: list[MachineState],
        health_check_fn: Callable[[MachineState], None],
    ):
        self.machines = machines
        self.health_check_fn = health_check_fn
        self._recovery_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def get_unhealthy_machines(self) -> list[MachineState]:
        return [m for m in self.machines if not m.healthy]

    async def start(self) -> None:
        if self._recovery_task is None or self._recovery_task.done():
            self._stop_event.clear()
            self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._recovery_task:
            try:
                await asyncio.wait_for(self._recovery_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._recovery_task.cancel()
            self._recovery_task = None

    async def _recovery_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.RECOVERY_CHECK_INTERVAL)

                unhealthy = self.get_unhealthy_machines()
                if not unhealthy:
                    continue

                for machine in unhealthy:
                    if self._stop_event.is_set():
                        break

                    if (
                        machine._consecutive_failures
                        >= self.MAX_FAILURES_BEFORE_BACKOFF
                    ):
                        time_since = time.time() - machine._last_failure_time
                        backoff = (
                            self.RECOVERY_CHECK_INTERVAL
                            * self.BACKOFF_MULTIPLIER
                            * (
                                machine._consecutive_failures
                                - self.MAX_FAILURES_BEFORE_BACKOFF
                                + 1
                            )
                        )
                        if time_since < backoff:
                            continue

                    machine._recovery_in_progress = True
                    try:
                        await asyncio.to_thread(self.health_check_fn, machine)
                        if machine.healthy:
                            machine._consecutive_failures = 0
                    except Exception:
                        machine._consecutive_failures += 1
                        machine._last_failure_time = time.time()
                    finally:
                        machine._recovery_in_progress = False
            except asyncio.CancelledError:
                break
            except Exception:
                pass


class MachineScheduler:
    """Scheduler for distributing chat requests across machines.

    Uses round-robin and capacity-based distribution for chat
    completion requests across multiple vLLM machines.
    """

    def __init__(self, machines: list[MachineState]):
        self.machines = machines
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    def get_healthy_machines(self) -> list[MachineState]:
        return [m for m in self.machines if m.healthy]

    def get_idle_machine(self) -> Optional[MachineState]:
        idle = [m for m in self.machines if m.healthy and m.is_idle]
        if not idle:
            self._idle_event.clear()
            return None
        idle.sort(key=lambda m: m.available_slots, reverse=True)
        return idle[0]

    def signal_idle(self) -> None:
        self._idle_event.set()

    def get_total_capacity(self, healthy: list[MachineState] = None) -> float:
        if healthy is None:
            healthy = self.get_healthy_machines()
        return sum(m.capacity for m in healthy)


@dataclass
class ClientsHealthResponse:
    """Health response for the multi-machine QVL clients."""

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


class _QVLClientsPipeline:
    """Core pipeline for distributing chat requests across machines.

    Distributes individual chat completion requests across multiple
    vLLM machines using capacity-based scheduling.
    """

    def __init__(
        self,
        machine_scheduler: MachineScheduler,
        on_progress: Optional[Callable[[int, int, float, dict], None]] = None,
        on_complete: Optional[Callable[[int, int, float], None]] = None,
    ):
        self.scheduler = machine_scheduler
        self.on_progress = on_progress
        self.on_complete = on_complete

    def run_pipeline(
        self,
        requests: list[dict],
        healthy: list[MachineState],
        request_fn: Callable,
        action_name: str = "chat",
        close_clients: bool = True,
    ) -> list[ChatResponse]:
        """Execute async pipeline distributing requests across machines.

        Args:
            requests: List of chat request dicts (each has 'messages', params)
            healthy: List of healthy machines
            request_fn: Async function (machine, request_dict) -> ChatResponse
            action_name: Name for logging
            close_clients: Whether to close async clients after completion

        Returns:
            List of ChatResponse in input order
        """
        for m in healthy:
            if m.async_client:
                m.async_client.reset()

        results = asyncio.run(
            self._run_pipeline_async(
                requests=requests,
                healthy=healthy,
                request_fn=request_fn,
                action_name=action_name,
                close_clients=close_clients,
            )
        )
        return results

    async def _run_pipeline_async(
        self,
        requests: list[dict],
        healthy: list[MachineState],
        request_fn: Callable,
        action_name: str = "chat",
        close_clients: bool = True,
    ) -> list[ChatResponse]:
        """Internal async pipeline implementation."""
        results_map: dict[int, ChatResponse] = {}
        pending_tasks: set[asyncio.Task] = set()
        errors: list[tuple[str, Exception]] = []
        request_queue = list(enumerate(requests))  # (index, request_dict) pairs
        queue_idx = 0

        machine_stats: dict[str, dict] = {
            m.endpoint: {"items": 0, "host": m.endpoint.split("//")[-1].split(":")[0]}
            for m in healthy
        }

        async def process_request(machine: MachineState, req_idx: int, req_dict: dict):
            task_start = time.perf_counter()
            try:
                result = await request_fn(machine, req_dict)
                return (
                    machine,
                    req_idx,
                    result,
                    time.perf_counter() - task_start,
                    None,
                )
            except Exception as e:
                return (machine, req_idx, None, time.perf_counter() - task_start, e)

        def dispatch_next(machine: MachineState) -> asyncio.Task | None:
            nonlocal queue_idx
            if queue_idx >= len(request_queue):
                return None
            req_idx, req_dict = request_queue[queue_idx]
            queue_idx += 1
            machine.mark_busy()
            return asyncio.create_task(process_request(machine, req_idx, req_dict))

        def dispatch_to_idle():
            tasks = []
            while queue_idx < len(request_queue):
                machine = self.scheduler.get_idle_machine()
                if machine is None:
                    break
                task = dispatch_next(machine)
                if task:
                    tasks.append(task)
                else:
                    break
            return tasks

        session_start = time.perf_counter()
        total_processed = 0
        last_log_time = 0.0

        while queue_idx < len(request_queue) or pending_tasks:
            if queue_idx < len(request_queue):
                new_tasks = dispatch_to_idle()
                pending_tasks.update(new_tasks)

            if pending_tasks:
                await asyncio.sleep(0)

            if not pending_tasks:
                break

            done, pending_tasks = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                machine, req_idx, result, latency, error = task.result()
                machine.mark_idle()
                self.scheduler.signal_idle()

                if error is None and result is not None:
                    results_map[req_idx] = result
                    machine_stats[machine.endpoint]["items"] += 1
                    machine.record_success()
                    total_processed += 1
                else:
                    machine.record_failure()
                    errors.append((machine.endpoint, error or Exception("Unknown")))

            # Dispatch more after completion
            if queue_idx < len(request_queue):
                new_tasks = dispatch_to_idle()
                pending_tasks.update(new_tasks)

            # Progress callback
            if self.on_progress and len(requests) >= 10:
                elapsed = time.perf_counter() - session_start
                if elapsed - last_log_time >= 5.0:
                    self.on_progress(
                        total_processed, len(requests), elapsed, machine_stats
                    )
                    last_log_time = elapsed

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
            combined.append(results_map[idx])

        if self.on_complete:
            self.on_complete(len(combined), len(combined), total_time)

        return combined


class _QVLClientsBase(ABC):
    """Abstract base class for multi-machine QVL clients.

    Subclasses need to:
    1. Set self._verbose before calling super().__init__()
    2. Initialize self._pipeline with appropriate callbacks
    """

    def __init__(self, endpoints: list[str]):
        self.endpoints = [ep.rstrip("/") for ep in endpoints]

        verbose = getattr(self, "_verbose", False)
        self.clients: list[QVLClient] = [
            QVLClient(endpoint=ep, verbose=verbose) for ep in self.endpoints
        ]
        self.async_clients: list[AsyncQVLClient] = [
            AsyncQVLClient(endpoint=ep, verbose=verbose) for ep in self.endpoints
        ]
        self.machines: list[MachineState] = [
            MachineState(endpoint=ep, client=sync_client, async_client=async_client)
            for ep, sync_client, async_client in zip(
                self.endpoints, self.clients, self.async_clients
            )
        ]

        for machine in self.machines:
            self._refresh_machine_health(machine)

        self._load_config()

        self.machine_scheduler = MachineScheduler(self.machines)
        self._health_recovery = HealthRecoveryManager(
            self.machines, self._refresh_machine_health
        )
        self._pipeline: Optional[_QVLClientsPipeline] = None
        self._rr_index = 0

    def _load_config(self) -> None:
        config = ExplorationConfig()
        for machine in self.machines:
            saved = config.get_machine_config(self.endpoints, machine.endpoint)
            if saved:
                self._apply_machine_config(machine, saved)

    def _apply_machine_config(self, machine: MachineState, saved: dict) -> None:
        config_instances = saved.get("instances", 0)
        optimal_max_concurrent = saved.get(
            "optimal_max_concurrent", machine._max_concurrent
        )
        config_throughput = saved.get("throughput", 0.0)

        machine._config_throughput = config_throughput
        machine._config_instances = config_instances

        if config_instances > 0 and machine.healthy_instances > 0:
            if machine.healthy_instances != config_instances:
                scale = machine.healthy_instances / config_instances
                machine._max_concurrent = max(1, int(optimal_max_concurrent * scale))
            else:
                machine._max_concurrent = optimal_max_concurrent
        else:
            machine._max_concurrent = optimal_max_concurrent

    def close(self) -> None:
        for client in self.clients:
            client.close()

    async def aclose(self) -> None:
        await self._health_recovery.stop()
        for async_client in self.async_clients:
            await async_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def refresh_health(self) -> ClientsHealthResponse:
        for machine in self.machines:
            self._refresh_machine_health(machine)
        return ClientsHealthResponse.from_machines(self.machines)

    def _refresh_machine_health(self, machine: MachineState) -> None:
        try:
            health = machine.client.health()
            machine.healthy = health.status == "healthy" or health.healthy > 0
            machine.healthy_instances = health.healthy
            machine.total_instances = health.total
        except Exception:
            machine.healthy = False
            machine.healthy_instances = 0

    def health(self) -> ClientsHealthResponse:
        return self.refresh_health()

    def _ensure_healthy(self) -> list[MachineState]:
        healthy = self.machine_scheduler.get_healthy_machines()
        if not healthy:
            self.refresh_health()
            healthy = self.machine_scheduler.get_healthy_machines()
        if not healthy:
            raise ValueError("No healthy machines available")
        return healthy

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> ChatResponse:
        """Send a single chat completion to a machine (round-robin)."""
        healthy = self._ensure_healthy()
        machine = healthy[self._rr_index % len(healthy)]
        self._rr_index += 1
        return machine.client.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        model: str = "",
    ) -> str:
        """Convenience method for single generation."""
        healthy = self._ensure_healthy()
        machine = healthy[self._rr_index % len(healthy)]
        self._rr_index += 1
        return machine.client.generate(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )

    def chat_batch(
        self,
        requests: list[dict],
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> list[ChatResponse]:
        """Send multiple chat completions distributed across machines.

        Args:
            requests: List of dicts, each with 'messages' key and optional params
            model: Default model name
            max_tokens: Default max tokens
            temperature: Default temperature

        Returns:
            List of ChatResponse in input order
        """
        if not requests:
            return []

        healthy = self._ensure_healthy()

        # Single request: use single machine
        if len(requests) == 1:
            machine = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            req = requests[0]
            resp = machine.client.chat(
                messages=req["messages"],
                model=req.get("model", model),
                max_tokens=req.get("max_tokens", max_tokens),
                temperature=req.get("temperature", temperature),
            )
            return [resp]

        # Enrich request dicts with defaults
        enriched = []
        for req in requests:
            enriched.append(
                {
                    "messages": req["messages"],
                    "model": req.get("model", model),
                    "max_tokens": req.get("max_tokens", max_tokens),
                    "temperature": req.get("temperature", temperature),
                }
            )

        # Pipeline distribution
        async def request_fn(machine: MachineState, req_dict: dict) -> ChatResponse:
            return await machine.async_client.chat(
                messages=req_dict["messages"],
                model=req_dict.get("model", ""),
                max_tokens=req_dict.get("max_tokens", 512),
                temperature=req_dict.get("temperature", 0.7),
            )

        return self._pipeline.run_pipeline(
            requests=enriched,
            healthy=healthy,
            request_fn=request_fn,
            action_name="chat_batch",
        )

    def generate_batch(
        self,
        prompts: list[str],
        images_list: list[list[str] | None] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        model: str = "",
    ) -> list[str]:
        """Convenience method for batch generation.

        Args:
            prompts: List of text prompts
            images_list: Optional list of image lists (one per prompt)
            system_prompt: Optional system message for all prompts
            max_tokens: Max tokens per generation
            temperature: Sampling temperature
            model: Model name

        Returns:
            List of generated text strings
        """
        from .client import build_vision_messages

        requests = []
        for i, prompt in enumerate(prompts):
            images = images_list[i] if images_list and i < len(images_list) else None
            messages = build_vision_messages(
                prompt=prompt,
                images=images,
                system_prompt=system_prompt,
            )
            requests.append(
                {
                    "messages": messages,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )

        responses = self.chat_batch(requests)
        return [r.text for r in responses]

    def info(self) -> list:
        """Get models info from all machines."""
        responses = []
        for machine in self.machines:
            try:
                responses.append(machine.client.models())
            except Exception:
                pass
        return responses
