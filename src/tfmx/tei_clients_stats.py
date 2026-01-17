"""TEI Multi-Machine Client with Stats - Testing/Exploration Version

Multi-machine TEI client with verbose logging and performance statistics,
designed for testing, benchmarking, and exploration scenarios.

This version adds detailed logging, per-machine stats, and progress tracking
on top of the core TEIClients functionality using composition pattern.

For clean production use without stats overhead, use TEIClients from tei_clients.py.

Features:
- All production features from TEIClients
- Verbose logging with per-machine throughput tracking
- Progress indicators for large batches (10k+ items)
- Performance statistics and metrics
"""

import argparse
import httpx
import json
from typing import Union, Iterable
from tclogger import logger, logstr

from .tei_client import TEIClient, AsyncTEIClient, InfoResponse, TIMEOUT
from .tei_clients_core import (
    MachineState,
    MachineScheduler,
    ClientsHealthResponse,
    _TEIClientsPipeline,
)
from .tei_performance import ExplorationConfig


# ANCHOR[id=clients-stats-clis]
CLI_EPILOG = """
Examples:
  export TEI_EPS="http://localhost:28800,http://ai122:28800"
  
  # Note: -E/--endpoints and -v/--verbose must be placed BEFORE the subcommand
  tei_clients_stats -E $TEI_EPS -v health
  tei_clients_stats -E $TEI_EPS -v embed "Hello" "World"
  tei_clients_stats -E $TEI_EPS -v lsh "Hello, world"
"""


class TEIClientsWithStats:
    """Multi-machine TEI client with verbose logging and performance stats.

    This is the stats-enabled version for testing/benchmarking. It provides
    detailed progress logging, per-machine throughput tracking, and other
    performance metrics.

    Example:
        ```python
        clients = TEIClientsWithStats([
            "http://machine1:28800",
            "http://machine2:28800",
        ], verbose=True)

        # Will print detailed progress and stats
        embs = clients.embed(large_dataset)
        lsh_hashes = clients.lsh(texts, bitn=2048)

        clients.close()
        ```

    Context manager:
        ```python
        with TEIClientsWithStats(endpoints, verbose=True) as clients:
            embs = clients.embed(texts)
        ```
    """

    def __init__(
        self,
        endpoints: list[str],
        timeout: float = TIMEOUT,
        verbose: bool = False,
    ):
        """Initialize multi-machine TEI client with stats.

        Args:
            endpoints: List of tei_machine endpoint URLs
                      (e.g., ["http://machine1:28800", "http://machine2:28800"])
            timeout: Request timeout in seconds (default: 60.0)
            verbose: Enable verbose logging and progress indicators
        """
        self.endpoints = [ep.rstrip("/") for ep in endpoints]
        self.timeout = timeout
        self.verbose = verbose

        # Create underlying clients for each endpoint
        self.clients: list[TEIClient] = [
            TEIClient(endpoint=ep, timeout=timeout, verbose=verbose)
            for ep in self.endpoints
        ]

        # Create async clients for pipeline
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

        # Load optimal batch sizes from config if available
        self._load_config()

        # Pipeline scheduler
        self.machine_scheduler = MachineScheduler(self.machines)

        # Pipeline executor with stats callbacks (composition)
        self._pipeline = _TEIClientsPipeline(
            machine_scheduler=self.machine_scheduler,
            on_progress=self._log_progress,
            on_complete=self._log_complete,
        )

        # Round-robin index for small batches
        self._rr_index = 0

    def _load_config(self) -> None:
        """Load optimal configurations from saved config file."""
        config = ExplorationConfig()
        for machine in self.machines:
            saved = config.get_machine_config(self.endpoints, machine.endpoint)
            if saved:
                machine.batch_size = saved.get("optimal_batch_size", machine.batch_size)
                machine._max_concurrent = saved.get(
                    "optimal_max_concurrent", machine._max_concurrent
                )
                if self.verbose:
                    short_name = machine.endpoint.split("//")[-1].split(":")[0]
                    logger.note(
                        f"[{short_name}] Loaded config: "
                        f"batch_size={machine.batch_size}, "
                        f"max_concurrent={machine._max_concurrent}"
                    )

    def _log_progress(
        self, processed: int, total: int, elapsed: float, machine_stats: dict
    ) -> None:
        """Callback for logging progress during pipeline execution.

        Format: [20%] 20000/100000 | localhost:1000/s | ai122:2400/s | 3400/s
        """
        pct = int(processed / total * 100)
        total_rate = processed / elapsed if elapsed > 0 else 0

        # Build per-machine stats: host:rate/s
        ep_stats = " | ".join(
            (
                f"{s['host']}:{int(s['items']/elapsed)}/s"
                if elapsed > 0
                else f"{s['host']}:0/s"
            )
            for s in machine_stats.values()
        )

        logger.mesg(
            f"  [{pct:3d}%] {processed:,}/{total:,} | {ep_stats} | {logstr.okay(int(total_rate))}/s"
        )

    def _log_complete(
        self, total_items: int, batch_count: int, total_time: float
    ) -> None:
        """Callback for logging completion stats."""
        throughput = total_items / total_time if total_time > 0 else 0
        logger.okay(
            f"[Pipeline] Complete: {total_items} items, {batch_count} batches, "
            f"{total_time:.2f}s, {throughput:.0f}/s"
        )

    def close(self) -> None:
        """Close all HTTP clients."""
        for client in self.clients:
            client.close()

    async def aclose(self) -> None:
        """Close all async HTTP clients."""
        for async_client in self.async_clients:
            await async_client.close()

    def __enter__(self) -> "TEIClientsWithStats":
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


class TEIClientsWithStatsArgParser:
    """Argument parser for TEI Clients with Stats CLI."""

    def __init__(self):
        # Create main parser with common arguments at root level
        self.parser = argparse.ArgumentParser(
            description="TEI Clients with Stats - Multi-machine client with verbose logging",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )

        # Add common arguments to main parser
        self._add_common_arguments(self.parser)

        # Setup subcommands (they won't have these common arguments repeated)
        self._setup_subcommands()
        self.args = self.parser.parse_args()

    def _add_common_arguments(self, parser):
        """Add common arguments to a parser."""
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
            help="Enable verbose output and progress logging",
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


class TEIClientsWithStatsCLI:
    """CLI interface for TEI Clients with Stats operations."""

    def __init__(self, clients: TEIClientsWithStats):
        """Initialize CLI with TEI clients.

        Args:
            clients: TEIClientsWithStats instance to use for operations
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
    arg_parser = TEIClientsWithStatsArgParser()
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
    clients = TEIClientsWithStats(endpoints=endpoints, verbose=args.verbose)

    try:
        cli = TEIClientsWithStatsCLI(clients)
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

    # LINK: src/tfmx/tei_clients_stats.py#clients-stats-clis
