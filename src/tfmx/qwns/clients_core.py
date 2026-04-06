"""Shared infrastructure for multi-machine QWN clients."""

import asyncio
import time

from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Optional

from .client import AsyncQWNClient, ChatResponse, DEFAULT_MAX_TOKENS, QWNClient
from .client import StreamChatResult
from .client import build_text_messages
from .compose import MAX_CONCURRENT_REQUESTS
from .performance import ExplorationConfig


@dataclass
class MachineState:
    endpoint: str
    client: QWNClient = field(repr=False)
    async_client: AsyncQWNClient = field(default=None, repr=False)
    healthy: bool = False
    healthy_instances: int = 0
    total_instances: int = 0
    _active_requests: int = 0
    _max_concurrent: int = MAX_CONCURRENT_REQUESTS
    _consecutive_failures: int = 0
    _last_failure_time: float = 0.0
    _recovery_in_progress: bool = False
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
        return [machine for machine in self.machines if not machine.healthy]

    async def start(self) -> None:
        if self._recovery_task is None or self._recovery_task.done():
            self._stop_event.clear()
            self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._recovery_task is None:
            return
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
    def __init__(self, machines: list[MachineState]):
        self.machines = machines
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    def get_healthy_machines(self) -> list[MachineState]:
        return [machine for machine in self.machines if machine.healthy]

    def get_idle_machine(self) -> Optional[MachineState]:
        idle_machines = [
            machine for machine in self.machines if machine.healthy and machine.is_idle
        ]
        if not idle_machines:
            self._idle_event.clear()
            return None
        idle_machines.sort(key=lambda machine: machine.available_slots, reverse=True)
        return idle_machines[0]

    def signal_idle(self) -> None:
        self._idle_event.set()

    def get_total_capacity(self, healthy: list[MachineState] | None = None) -> float:
        healthy = healthy or self.get_healthy_machines()
        return sum(machine.capacity for machine in healthy)


@dataclass
class ClientsHealthResponse:
    status: str
    healthy_machines: int
    total_machines: int
    healthy_instances: int
    total_instances: int

    @classmethod
    def from_machines(cls, machines: list[MachineState]) -> "ClientsHealthResponse":
        healthy_machines = sum(1 for machine in machines if machine.healthy)
        healthy_instances = sum(machine.healthy_instances for machine in machines)
        total_instances = sum(machine.total_instances for machine in machines)
        return cls(
            status="healthy" if healthy_machines > 0 else "unhealthy",
            healthy_machines=healthy_machines,
            total_machines=len(machines),
            healthy_instances=healthy_instances,
            total_instances=total_instances,
        )


@dataclass
class BatchChatOutcome:
    response: ChatResponse | StreamChatResult | None = None
    error: Exception | None = None
    latency_sec: float = 0.0
    first_token_latency_sec: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.response is not None and self.error is None


class _QWNClientsPipeline:
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
        outcomes = self.run_pipeline_outcomes(
            requests=requests,
            healthy=healthy,
            request_fn=request_fn,
            action_name=action_name,
            close_clients=close_clients,
        )
        return [
            outcome.response for outcome in outcomes if outcome.response is not None
        ]

    def run_pipeline_outcomes(
        self,
        requests: list[dict],
        healthy: list[MachineState],
        request_fn: Callable,
        action_name: str = "chat",
        close_clients: bool = True,
    ) -> list[BatchChatOutcome]:
        for machine in healthy:
            machine.async_client.reset()
        return asyncio.run(
            self._run_pipeline_async(
                requests=requests,
                healthy=healthy,
                request_fn=request_fn,
                action_name=action_name,
                close_clients=close_clients,
            )
        )

    async def _run_pipeline_async(
        self,
        requests: list[dict],
        healthy: list[MachineState],
        request_fn: Callable,
        action_name: str,
        close_clients: bool,
    ) -> list[BatchChatOutcome]:
        total = len(requests)
        start_time = time.perf_counter()
        machine_stats = {
            machine.endpoint: {"host": machine.endpoint, "items": 0}
            for machine in healthy
        }
        results: list[BatchChatOutcome | None] = [None] * total
        tasks: set[asyncio.Task] = set()
        completed = 0
        next_index = 0

        async def submit(machine: MachineState, index: int, request: dict):
            machine.mark_busy()
            request_started_at = time.perf_counter()
            try:
                response = await request_fn(machine, request)
                latency_sec = time.perf_counter() - request_started_at
                setattr(response, "_latency_sec", latency_sec)
                return machine, index, response, None, latency_sec
            except Exception as exc:
                return (
                    machine,
                    index,
                    None,
                    exc,
                    time.perf_counter() - request_started_at,
                )
            finally:
                machine.mark_idle()
                self.scheduler.signal_idle()

        try:
            while next_index < total or tasks:
                while next_index < total:
                    machine = self.scheduler.get_idle_machine()
                    if machine is None:
                        break
                    task = asyncio.create_task(
                        submit(machine, next_index, requests[next_index])
                    )
                    tasks.add(task)
                    next_index += 1

                if not tasks:
                    await asyncio.sleep(0.01)
                    continue

                done, tasks = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for finished in done:
                    machine, index, response, error, latency_sec = await finished
                    if error is None and response is not None:
                        machine.record_success()
                        results[index] = BatchChatOutcome(
                            response=response,
                            latency_sec=latency_sec,
                            first_token_latency_sec=getattr(
                                response, "first_token_latency_sec", 0.0
                            ),
                        )
                        machine_stats[machine.endpoint]["items"] += 1
                    else:
                        machine.record_failure()
                        results[index] = BatchChatOutcome(
                            error=error,
                            latency_sec=latency_sec,
                        )
                    completed += 1
                    if self.on_progress:
                        elapsed = time.perf_counter() - start_time
                        self.on_progress(completed, total, elapsed, machine_stats)

            elapsed = time.perf_counter() - start_time
            final_results = [
                (
                    result
                    if result is not None
                    else BatchChatOutcome(error=RuntimeError("missing pipeline result"))
                )
                for result in results
            ]
            if self.on_complete:
                self.on_complete(
                    sum(1 for result in final_results if result.succeeded),
                    total,
                    elapsed,
                )
            return final_results
        finally:
            if close_clients:
                for machine in healthy:
                    await machine.async_client.close()


class _QWNClientsBase(ABC):
    def __init__(self, endpoints: list[str]):
        self.endpoints = [endpoint.rstrip("/") for endpoint in endpoints]
        verbose = getattr(self, "_verbose", False)
        self.clients = [
            QWNClient(endpoint=endpoint, verbose=verbose) for endpoint in self.endpoints
        ]
        self.async_clients = [
            AsyncQWNClient(endpoint=endpoint, verbose=verbose)
            for endpoint in self.endpoints
        ]
        self.machines = [
            MachineState(endpoint=endpoint, client=client, async_client=async_client)
            for endpoint, client, async_client in zip(
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
        self._pipeline: Optional[_QWNClientsPipeline] = None
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
            "optimal_max_concurrent",
            machine._max_concurrent,
        )
        config_throughput = saved.get("throughput", 0.0)

        machine._config_instances = config_instances
        machine._config_throughput = config_throughput

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
        for client in self.async_clients:
            await client.close()

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
            machine.total_instances = 0

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
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> ChatResponse:
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
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "",
    ) -> str:
        healthy = self._ensure_healthy()
        machine = healthy[self._rr_index % len(healthy)]
        self._rr_index += 1
        messages = build_text_messages(prompt=prompt, system_prompt=system_prompt)
        response = machine.client.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.text

    def chat_batch(
        self,
        requests: list[dict],
        model: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[ChatResponse]:
        outcomes = self.chat_batch_outcomes(
            requests=requests,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            measure_ttft=False,
        )
        return [
            outcome.response for outcome in outcomes if outcome.response is not None
        ]

    def chat_batch_outcomes(
        self,
        requests: list[dict],
        model: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        measure_ttft: bool = False,
    ) -> list[BatchChatOutcome]:
        if not requests:
            return []

        healthy = self._ensure_healthy()
        if len(requests) == 1:
            machine = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            request = requests[0]
            started_at = time.perf_counter()
            try:
                if measure_ttft:
                    response = machine.client.stream_chat(
                        messages=request["messages"],
                        model=request.get("model", model),
                        max_tokens=request.get("max_tokens", max_tokens),
                        temperature=request.get("temperature", temperature),
                        top_p=request.get("top_p", top_p),
                    )
                else:
                    response = machine.client.chat(
                        messages=request["messages"],
                        model=request.get("model", model),
                        max_tokens=request.get("max_tokens", max_tokens),
                        temperature=request.get("temperature", temperature),
                        top_p=request.get("top_p", top_p),
                    )
                latency_sec = time.perf_counter() - started_at
                setattr(response, "_latency_sec", latency_sec)
                return [
                    BatchChatOutcome(
                        response=response,
                        latency_sec=latency_sec,
                        first_token_latency_sec=getattr(
                            response, "first_token_latency_sec", 0.0
                        ),
                    )
                ]
            except Exception as exc:
                return [
                    BatchChatOutcome(
                        error=exc,
                        latency_sec=time.perf_counter() - started_at,
                    )
                ]

        enriched = []
        for request in requests:
            enriched.append(
                {
                    "messages": request["messages"],
                    "model": request.get("model", model),
                    "max_tokens": request.get("max_tokens", max_tokens),
                    "temperature": request.get("temperature", temperature),
                    "top_p": request.get("top_p", top_p),
                }
            )

        if measure_ttft:

            async def request_fn(
                machine: MachineState, request_dict: dict
            ) -> StreamChatResult:
                return await machine.async_client.stream_chat(
                    messages=request_dict["messages"],
                    model=request_dict.get("model", ""),
                    max_tokens=request_dict.get("max_tokens", 512),
                    temperature=request_dict.get("temperature", 0.7),
                    top_p=request_dict.get("top_p", 0.9),
                )

        else:

            async def request_fn(
                machine: MachineState, request_dict: dict
            ) -> ChatResponse:
                return await machine.async_client.chat(
                    messages=request_dict["messages"],
                    model=request_dict.get("model", ""),
                    max_tokens=request_dict.get("max_tokens", 512),
                    temperature=request_dict.get("temperature", 0.7),
                    top_p=request_dict.get("top_p", 0.9),
                )

        return self._pipeline.run_pipeline_outcomes(
            requests=enriched,
            healthy=healthy,
            request_fn=request_fn,
            action_name="chat_batch",
        )

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "",
    ) -> list[str]:
        requests = []
        for prompt in prompts:
            requests.append(
                {
                    "messages": build_text_messages(
                        prompt=prompt, system_prompt=system_prompt
                    ),
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        responses = self.chat_batch(requests)
        return [response.text for response in responses]

    def info(self) -> list:
        responses = []
        for machine in self.machines:
            try:
                responses.append(machine.client.info())
            except Exception:
                pass
        return responses
