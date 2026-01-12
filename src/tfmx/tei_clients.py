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
import httpx
import json

from dataclasses import dataclass
from tclogger import logger
from typing import Union, Optional

from .tei_client import TEIClient, InfoResponse, TIMEOUT
from .tei_compose import MAX_CLIENT_BATCH_SIZE
from .tei_scheduler import IdleFillingScheduler, distribute_with_scheduler


@dataclass
class MachineInfo:
    """Information about a single tei_machine endpoint."""

    endpoint: str
    healthy: bool = False
    healthy_instances: int = 0
    total_instances: int = 0

    # Throughput tracking (items per second) using EMA
    _throughput_ema: float = 0.0
    _ema_alpha: float = 0.3  # EMA smoothing factor
    _total_items: int = 0
    _total_latency: float = 0.0
    _total_requests: int = 0

    @property
    def weight(self) -> int:
        """Weight for load balancing based on healthy instances."""
        return self.healthy_instances if self.healthy else 0

    @property
    def throughput(self) -> float:
        """Get estimated throughput in items/second."""
        return self._throughput_ema

    def record_success(self, latency: float, n_items: int) -> None:
        """Record a successful request and update throughput estimate."""
        self._total_requests += 1
        self._total_items += n_items
        self._total_latency += latency

        # Update throughput EMA
        if latency > 0:
            current_throughput = n_items / latency
            if self._throughput_ema == 0:
                self._throughput_ema = current_throughput
            else:
                self._throughput_ema = (
                    self._ema_alpha * current_throughput
                    + (1 - self._ema_alpha) * self._throughput_ema
                )


@dataclass
class ClientsHealthResponse:
    """Health response for the multi-machine clients."""

    status: str
    healthy_machines: int
    total_machines: int
    healthy_instances: int
    total_instances: int

    @classmethod
    def from_machines(cls, machines: list[MachineInfo]) -> "ClientsHealthResponse":
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

        # Machine info for load balancing
        self.machines: list[MachineInfo] = [
            MachineInfo(endpoint=ep) for ep in self.endpoints
        ]

        # Round-robin index
        self._rr_index = 0

        # Idle-filling scheduler for simple load balancing
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

    def _get_throughput_ratios(
        self, healthy: list[tuple[int, "TEIClient", MachineInfo]]
    ) -> list[float]:
        """Calculate throughput ratios for healthy machines.

        Returns list of ratios (sums to 1.0) based on each machine's throughput.
        If no throughput data available, returns equal ratios.
        """
        throughputs = [m.throughput for _, _, m in healthy]
        total = sum(throughputs)

        if total > 0:
            return [t / total for t in throughputs]
        else:
            # No throughput data: equal distribution
            n = len(healthy)
            return [1.0 / n] * n

    def _split_by_throughput(
        self,
        inputs: list[str],
        healthy: list[tuple[int, "TEIClient", MachineInfo]],
    ) -> list[tuple[int, "TEIClient", MachineInfo, list[str], int, int]]:
        """Split inputs across machines proportionally to their throughput.

        Returns list of (idx, client, machine_info, chunk, start_idx, end_idx) tuples.
        """
        ratios = self._get_throughput_ratios(healthy)
        n_inputs = len(inputs)

        # Calculate chunk sizes based on ratios
        chunks_info = []
        start = 0
        remaining = n_inputs

        for i, (idx, client, machine) in enumerate(healthy):
            if i == len(healthy) - 1:
                # Last machine gets all remaining
                chunk_size = remaining
            else:
                # Proportional allocation
                chunk_size = int(n_inputs * ratios[i])
                chunk_size = min(chunk_size, remaining)  # Don't exceed remaining

            end = start + chunk_size
            chunk = inputs[start:end]

            if chunk:  # Only include non-empty chunks
                chunks_info.append((idx, client, machine, chunk, start, end))

            start = end
            remaining = n_inputs - start

        return chunks_info

    def refresh_health(self) -> ClientsHealthResponse:
        """Refresh health status of all machines.

        Returns:
            ClientsHealthResponse with aggregated health info.
        """
        for i, client in enumerate(self.clients):
            try:
                health = client.health()
                self.machines[i].healthy = (
                    health.status == "healthy" or health.healthy > 0
                )
                self.machines[i].healthy_instances = health.healthy
                self.machines[i].total_instances = health.total
            except Exception:
                self.machines[i].healthy = False
                self.machines[i].healthy_instances = 0

        return ClientsHealthResponse.from_machines(self.machines)

    def health(self) -> ClientsHealthResponse:
        """Check health status of all machines.

        Returns:
            ClientsHealthResponse with aggregated health info.
        """
        return self.refresh_health()

    def _get_healthy_clients(self) -> list[tuple[int, TEIClient, MachineInfo]]:
        """Get list of healthy clients with their indices and machine info."""
        return [
            (i, self.clients[i], self.machines[i])
            for i in range(len(self.clients))
            if self.machines[i].healthy
        ]

    def embed(
        self,
        inputs: Union[str, list[str]],
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for input texts using multiple machines.

        Distributes inputs across healthy machines using adaptive scheduling (EFT-based).

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

        # Refresh health if needed (first call or no healthy machines)
        healthy = self._get_healthy_clients()
        if not healthy:
            self.refresh_health()
            healthy = self._get_healthy_clients()

        if not healthy:
            raise ValueError("No healthy machines available")

        # For small inputs or single machine, use simple path
        if len(inputs) <= 10 or len(healthy) == 1:
            # Round-robin single request
            _, client, _ = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return client.embed(inputs, normalize=normalize, truncate=truncate)

        # Use adaptive scheduler for distribution
        return self._embed_with_scheduler(inputs, healthy, normalize, truncate)

    def _embed_with_scheduler(
        self,
        inputs: list[str],
        healthy: list[tuple[int, TEIClient, MachineInfo]],
        normalize: bool,
        truncate: bool,
    ) -> list[list[float]]:
        """Distribute embed requests across machines based on throughput ratios.

        Strategy:
        1. Split inputs proportionally to each machine's throughput
        2. Send requests to all machines in parallel
        3. Update throughput estimates based on actual performance
        4. Combine results in original order
        """
        import asyncio
        import time

        # Split inputs based on throughput ratios
        chunks_info = self._split_by_throughput(inputs, healthy)

        # Define async task for each machine
        async def process_chunk(
            client: TEIClient, machine: MachineInfo, chunk: list[str]
        ) -> tuple[list[list[float]], float, int, MachineInfo, Exception | None]:
            """Process a chunk on a machine."""
            start_time = time.perf_counter()
            try:
                results = await asyncio.to_thread(
                    client.embed, chunk, normalize=normalize, truncate=truncate
                )
                latency = time.perf_counter() - start_time
                return (results, latency, len(chunk), machine, None)
            except Exception as e:
                latency = time.perf_counter() - start_time
                return ([], latency, len(chunk), machine, e)

        # Run all requests in parallel
        async def run_parallel():
            tasks = [
                process_chunk(client, machine, chunk)
                for _, client, machine, chunk, _, _ in chunks_info
            ]
            return await asyncio.gather(*tasks)

        results_list = asyncio.run(run_parallel())

        # Process results: update throughput and collect outputs
        all_results: list[tuple[int, int, list[list[float]]]] = []
        errors = []

        for i, (results, latency, n_items, machine, error) in enumerate(results_list):
            _, _, _, _, start_idx, end_idx = chunks_info[i]

            if error is None:
                machine.record_success(latency, n_items)
                all_results.append((start_idx, end_idx, results))
            else:
                machine.healthy = False
                errors.append((machine.endpoint, error))

        if not all_results:
            raise ValueError(f"All requests failed: {errors}")

        # Sort by start_idx and combine results
        all_results.sort(key=lambda x: x[0])
        combined = []
        for start_idx, end_idx, results in all_results:
            combined.extend(results)

        self._log_okay(
            "embed",
            f"n={len(combined)}, machines={len(all_results)}",
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

        Distributes inputs across healthy machines using idle-filling scheduling.

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
        healthy = self._get_healthy_clients()
        if not healthy:
            self.refresh_health()
            healthy = self._get_healthy_clients()

        if not healthy:
            raise ValueError("No healthy machines available")

        # For very small inputs, use simple path
        if len(inputs) <= 10:
            _, client, _ = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return client.lsh(inputs, bitn=bitn, normalize=normalize, truncate=truncate)

        # Single machine optimization: send directly without scheduler splitting
        # This lets the TEIMachine's internal pipeline handle distribution
        if len(healthy) == 1:
            _, client, _ = healthy[0]
            return client.lsh(inputs, bitn=bitn, normalize=normalize, truncate=truncate)

        # Multiple machines: use scheduler for distribution
        return self._lsh_with_scheduler(inputs, healthy, bitn, normalize, truncate)

    def _lsh_with_scheduler(
        self,
        inputs: list[str],
        healthy: list[tuple[int, TEIClient, MachineInfo]],
        bitn: int,
        normalize: bool,
        truncate: bool,
    ) -> list[str]:
        """Distribute LSH requests across machines based on throughput ratios.

        Strategy:
        1. Split inputs proportionally to each machine's throughput
        2. Send requests to all machines in parallel
        3. Update throughput estimates based on actual performance
        4. Combine results in original order
        """
        import asyncio
        import time

        # Split inputs based on throughput ratios
        chunks_info = self._split_by_throughput(inputs, healthy)

        if self.verbose:
            ratios = self._get_throughput_ratios(healthy)
            logger.mesg(
                f"[TEIClients] Distributing {len(inputs)} items across {len(healthy)} machines:"
            )
            for i, (_, _, m, chunk, _, _) in enumerate(chunks_info):
                logger.mesg(
                    f"  {m.endpoint}: {len(chunk)} items "
                    f"({ratios[i]*100:.1f}%, throughput={m.throughput:.0f}/s)"
                )

        # Define async task for each machine
        async def process_chunk(
            client: TEIClient, machine: MachineInfo, chunk: list[str]
        ) -> tuple[list[str], float, int, MachineInfo, Exception | None]:
            """Process a chunk on a machine, return (results, latency, n_items, machine, error)."""
            start_time = time.perf_counter()
            try:
                results = await asyncio.to_thread(
                    client.lsh, chunk, bitn=bitn, normalize=normalize, truncate=truncate
                )
                latency = time.perf_counter() - start_time
                return (results, latency, len(chunk), machine, None)
            except Exception as e:
                latency = time.perf_counter() - start_time
                return ([], latency, len(chunk), machine, e)

        # Run all requests in parallel
        async def run_parallel():
            tasks = [
                process_chunk(client, machine, chunk)
                for _, client, machine, chunk, _, _ in chunks_info
            ]
            return await asyncio.gather(*tasks)

        results_list = asyncio.run(run_parallel())

        # Process results: update throughput and collect outputs
        all_results: list[tuple[int, int, list[str]]] = (
            []
        )  # (start_idx, end_idx, results)
        errors = []

        for i, (results, latency, n_items, machine, error) in enumerate(results_list):
            _, _, _, _, start_idx, end_idx = chunks_info[i]

            if error is None:
                # Update throughput estimate
                machine.record_success(latency, n_items)
                all_results.append((start_idx, end_idx, results))

                if self.verbose:
                    throughput = n_items / latency if latency > 0 else 0
                    logger.mesg(
                        f"  {machine.endpoint}: {n_items} items in {latency*1000:.0f}ms "
                        f"({throughput:.0f}/s, EMA={machine.throughput:.0f}/s)"
                    )
            else:
                machine.healthy = False
                errors.append((machine.endpoint, error))

        # Check for complete failure
        if not all_results:
            raise ValueError(f"All requests failed: {errors}")

        # Sort by start_idx and combine results
        all_results.sort(key=lambda x: x[0])
        combined = []
        for start_idx, end_idx, results in all_results:
            combined.extend(results)

        self._log_okay(
            "lsh",
            f"n={len(combined)}, bitn={bitn}, machines={len(all_results)}",
        )

        return combined

    def get_scheduler_stats(self) -> dict:
        """Get scheduler statistics."""
        return self.scheduler.get_stats_summary()

    def get_machine_stats(self) -> list[dict]:
        """Get throughput statistics for all machines."""
        return [
            {
                "endpoint": m.endpoint,
                "healthy": m.healthy,
                "throughput": m.throughput,
                "total_items": m._total_items,
                "total_requests": m._total_requests,
            }
            for m in self.machines
        ]

    def info(self) -> list[InfoResponse]:
        """Get info from all machines.

        Returns:
            List of InfoResponse from each machine.
        """
        responses = []
        for client in self.clients:
            try:
                responses.append(client.info())
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
        clients = self.clients.clients
        if not clients:
            logger.warn("× No machine info available")
            return

        for i, client in enumerate(clients):
            logger.note(f"[Machine {i+1}] {self.clients.endpoints[i]}")
            client.log_machine_health()

    def run_info(self) -> None:
        """Get and display info from all machines."""
        clients = self.clients.clients
        if not clients:
            logger.warn("× No machine info available")
            return

        for i, client in enumerate(clients):
            logger.okay(f"[Machine {i+1}] {self.clients.endpoints[i]}")
            client.log_machine_info()
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
