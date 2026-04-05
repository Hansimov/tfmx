"""Verbose multi-machine QWN client with progress logging."""

import sys
import time

from .clients_core import ClientsHealthResponse, MachineState, _QWNClientsBase
from .clients_core import _QWNClientsPipeline


def _log(message: str, file=sys.stderr, **kwargs) -> None:
    print(message, file=file, flush=True, **kwargs)


class QWNClientsWithStats(_QWNClientsBase):
    def __init__(self, endpoints: list[str], verbose: bool = True):
        self._verbose = verbose
        super().__init__(endpoints)
        self._pipeline = _QWNClientsPipeline(
            machine_scheduler=self.machine_scheduler,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
        )

    def __repr__(self) -> str:
        healthy = sum(1 for machine in self.machines if machine.healthy)
        return f"QWNClientsWithStats(machines={len(self.machines)}, healthy={healthy})"

    def _on_progress(
        self,
        completed: int,
        total: int,
        elapsed: float,
        machine_stats: dict,
    ) -> None:
        if not self._verbose:
            return
        percent = completed / total * 100 if total > 0 else 0
        rps = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rps if rps > 0 else 0
        _log(
            f"  [{completed}/{total}] {percent:.0f}% | {rps:.1f} req/s | ETA {eta:.0f}s"
        )
        if machine_stats:
            stats_text = ", ".join(
                f"{stats['host']}:{stats['items']}"
                for stats in machine_stats.values()
                if stats["items"] > 0
            )
            if stats_text:
                _log(f"    machines: {stats_text}")

    def _on_complete(
        self, total_responses: int, total_requests: int, total_time: float
    ) -> None:
        if not self._verbose:
            return
        rps = total_responses / total_time if total_time > 0 else 0
        avg_latency = total_time / total_responses if total_responses > 0 else 0
        _log("\n  Session complete:")
        _log(f"    Responses: {total_responses}/{total_requests}")
        _log(f"    Time: {total_time:.2f}s")
        _log(f"    Throughput: {rps:.2f} req/s")
        _log(f"    Avg latency: {avg_latency:.3f}s per request")

    def refresh_health(self) -> ClientsHealthResponse:
        if self._verbose:
            _log("Checking health of all machines...")
        result = super().refresh_health()
        if self._verbose:
            _log(
                f"  {result.status}: {result.healthy_machines}/{result.total_machines} machines healthy"
            )
            for machine in self.machines:
                status = "OK" if machine.healthy else "FAIL"
                _log(
                    f"    [{status}] {machine.endpoint} "
                    f"({machine.healthy_instances}/{machine.total_instances} instances)"
                )
        return result

    def chat(self, messages: list[dict], **kwargs):
        if self._verbose:
            preview = ""
            for message in messages:
                if message.get("role") == "user":
                    preview = str(message.get("content", ""))[:60]
                    break
            _log(f"Chat: {preview}...")

        started_at = time.perf_counter()
        result = super().chat(messages, **kwargs)
        latency = time.perf_counter() - started_at
        if self._verbose:
            _log(
                f"  -> {result.usage.completion_tokens} tokens in {latency:.2f}s "
                f"({result.usage.completion_tokens / latency:.1f} tok/s)"
            )
        return result

    def chat_batch(self, requests: list[dict], **kwargs):
        if self._verbose:
            healthy = self.machine_scheduler.get_healthy_machines()
            total_capacity = sum(machine._max_concurrent for machine in healthy)
            _log(f"\nBatch chat: {len(requests)} requests")
            _log(f"  Healthy machines: {len(healthy)}")
            _log(f"  Total capacity: {total_capacity} concurrent requests")

        started_at = time.perf_counter()
        result = super().chat_batch(requests, **kwargs)
        total_time = time.perf_counter() - started_at
        if self._verbose:
            total_gen_tokens = sum(
                response.usage.completion_tokens for response in result
            )
            total_prompt_tokens = sum(
                response.usage.prompt_tokens for response in result
            )
            _log("\n  Batch summary:")
            _log(f"    Prompt tokens: {total_prompt_tokens}")
            _log(f"    Generated tokens: {total_gen_tokens}")
            _log(
                f"    Throughput: {total_gen_tokens / total_time:.1f} tok/s "
                f"({len(result) / total_time:.2f} req/s)"
            )
        return result
