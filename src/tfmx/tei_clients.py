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

    @property
    def weight(self) -> int:
        """Weight for load balancing based on healthy instances."""
        return self.healthy_instances if self.healthy else 0


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
        """Distribute embed requests using idle-filling scheduler."""
        # Update scheduler with healthy clients
        healthy_clients = [client for _, client, _ in healthy]
        self.scheduler.update_workers(healthy_clients)

        # Define async process function
        async def process_on_client(
            client: TEIClient, chunk: list[str]
        ) -> list[list[float]]:
            return client.embed(chunk, normalize=normalize, truncate=truncate)

        # Use scheduler distribution with asyncio
        import asyncio

        embeddings, details = asyncio.run(
            distribute_with_scheduler(
                scheduler=self.scheduler,
                inputs=inputs,
                process_func=process_on_client,
            )
        )

        # Update machine health on errors
        for d in details:
            if not d.success:
                for m in self.machines:
                    if m.endpoint == d.worker_id:
                        m.healthy = False
                        break

        # Check for complete failure
        successful = [d for d in details if d.success]
        if not successful:
            errors = [(d.worker_id, d.error) for d in details if not d.success]
            raise ValueError(f"All requests failed: {errors}")

        self._log_okay(
            "embed",
            f"n={len(embeddings)}, machines={len(set(d.worker_id for d in successful))}",
        )

        return embeddings

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
        """Distribute LSH requests using idle-filling scheduler."""
        # Update scheduler with healthy clients
        healthy_clients = [client for _, client, _ in healthy]
        self.scheduler.update_workers(healthy_clients)

        # Define async process function
        async def process_on_client(client: TEIClient, chunk: list[str]) -> list[str]:
            return client.lsh(chunk, bitn=bitn, normalize=normalize, truncate=truncate)

        # Use scheduler distribution with asyncio
        import asyncio

        hashes, details = asyncio.run(
            distribute_with_scheduler(
                scheduler=self.scheduler,
                inputs=inputs,
                process_func=process_on_client,
            )
        )

        # Update machine health on errors
        for d in details:
            if not d.success:
                for m in self.machines:
                    if m.endpoint == d.worker_id:
                        m.healthy = False
                        break

        # Check for complete failure
        successful = [d for d in details if d.success]
        if not successful:
            errors = [(d.worker_id, d.error) for d in details if not d.success]
            raise ValueError(f"All requests failed: {errors}")

        self._log_okay(
            "lsh",
            f"n={len(hashes)}, bitn={bitn}, machines={len(set(d.worker_id for d in successful))}",
        )

        return hashes

    def get_scheduler_stats(self) -> dict:
        """Get scheduler statistics."""
        return self.scheduler.get_stats_summary()

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
