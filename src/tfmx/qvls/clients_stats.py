"""QVL Clients with Stats - Verbose Multi-Machine Client with Logging

A wrapper around QVLClients that adds detailed per-request logging,
per-machine throughput stats, and progress reporting.

Usage:
    from tfmx.qvls.clients_stats import QVLClientsWithStats

    clients = QVLClientsWithStats(["http://host1:29800", "http://host2:29800"])
    responses = clients.chat_batch([{"messages": [{"role": "user", "content": "Hi"}]}])
    # Logs: progress updates, per-machine stats, throughput summary
"""

import sys
import time
from typing import Optional

from .clients_core import (
    _QVLClientsBase,
    _QVLClientsPipeline,
    ClientsHealthResponse,
    MachineState,
)


def _log(msg: str, file=sys.stderr, **kwargs) -> None:
    print(msg, file=file, flush=True, **kwargs)


class QVLClientsWithStats(_QVLClientsBase):
    """Multi-machine QVL client with verbose logging and stats.

    Logs per-request progress, per-machine throughput, and
    session summaries during batch operations.
    """

    def __init__(self, endpoints: list[str], verbose: bool = True):
        self._verbose = verbose
        super().__init__(endpoints)
        self._pipeline = _QVLClientsPipeline(
            machine_scheduler=self.machine_scheduler,
            on_progress=self._on_progress,
            on_complete=self._on_complete,
        )
        self._session_start: float = 0.0
        self._total_tokens: int = 0

    def __repr__(self) -> str:
        healthy = sum(1 for m in self.machines if m.healthy)
        return (
            f"QVLClientsWithStats(machines={len(self.machines)}, " f"healthy={healthy})"
        )

    def _on_progress(
        self,
        completed: int,
        total: int,
        elapsed: float,
        machine_stats: dict,
    ) -> None:
        if not self._verbose:
            return

        pct = completed / total * 100 if total > 0 else 0
        rps = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rps if rps > 0 else 0

        _log(
            f"  [{completed}/{total}] {pct:.0f}% " f"| {rps:.1f} req/s | ETA {eta:.0f}s"
        )

        if machine_stats:
            stats_str = ", ".join(
                f"{v['host']}:{v['items']}"
                for v in machine_stats.values()
                if v["items"] > 0
            )
            if stats_str:
                _log(f"    machines: {stats_str}")

    def _on_complete(
        self,
        total_responses: int,
        total_requests: int,
        total_time: float,
    ) -> None:
        if not self._verbose:
            return

        rps = total_responses / total_time if total_time > 0 else 0
        avg_latency = total_time / total_responses if total_responses > 0 else 0

        _log(f"\n  Session complete:")
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
            content_preview = ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        content_preview = content[:60]
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                content_preview = part.get("text", "")[:60]
                                break
                        if not content_preview:
                            content_preview = f"[{len(content)} content parts]"
                    break
            _log(f"Chat: {content_preview}...")

        t0 = time.perf_counter()
        result = super().chat(messages, **kwargs)
        latency = time.perf_counter() - t0

        if self._verbose:
            _log(
                f"  -> {result.usage.completion_tokens} tokens in {latency:.2f}s "
                f"({result.usage.completion_tokens / latency:.1f} tok/s)"
            )

        return result

    def chat_batch(self, requests: list[dict], **kwargs):
        if self._verbose:
            _log(f"\nBatch chat: {len(requests)} requests")
            healthy = self.machine_scheduler.get_healthy_machines()
            _log(f"  Healthy machines: {len(healthy)}")
            total_capacity = sum(m._max_concurrent for m in healthy)
            _log(f"  Total capacity: {total_capacity} concurrent requests")

        t0 = time.perf_counter()
        result = super().chat_batch(requests, **kwargs)
        total_time = time.perf_counter() - t0

        if self._verbose:
            total_gen_tokens = sum(r.usage.completion_tokens for r in result)
            total_prompt_tokens = sum(r.usage.prompt_tokens for r in result)
            _log(f"\n  Batch summary:")
            _log(f"    Prompt tokens: {total_prompt_tokens}")
            _log(f"    Generated tokens: {total_gen_tokens}")
            _log(
                f"    Throughput: {total_gen_tokens / total_time:.1f} tok/s "
                f"({len(result) / total_time:.2f} req/s)"
            )

        return result


class QVLClientsStatsArgParser:
    """Argument parser that creates QVLClientsWithStats."""

    def create_client(self, args) -> QVLClientsWithStats:
        return QVLClientsWithStats(
            endpoints=args.endpoints,
            verbose=True,
        )


def run_cli_main(args=None) -> None:
    """Entry point for qvl_clients_stats CLI."""
    from .clients_cli import QVLClientsArgParserBase, QVLClientsCLIBase

    class StatsArgParser(QVLClientsArgParserBase):
        def create_client(self, parsed_args) -> QVLClientsWithStats:
            return QVLClientsWithStats(
                endpoints=parsed_args.endpoints,
                verbose=True,
            )

    class StatsCLI(QVLClientsCLIBase):
        def __init__(self):
            super().__init__(StatsArgParser("QVL Clients CLI (with stats)"))

    cli = StatsCLI()
    cli.run(args)


if __name__ == "__main__":
    run_cli_main()
