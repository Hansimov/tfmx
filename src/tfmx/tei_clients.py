"""TEI (Text Embeddings Inference) Multi-Machine Client

This module provides a client for connecting to multiple TEI machines,
with client-side load balancing across machines.
"""

# ANCHOR[id=clients-clis]
CLI_EPILOG = """
Examples:
  export TEI_EPS="http://localhost:28800,http://ai122:28800"
  
  tei_clients health -E $TEI_EPS
  tei_clients info -E $TEI_EPS
  tei_clients embed -E $TEI_EPS "Hello" "World"
  tei_clients lsh -E $TEI_EPS "Hello"
  tei_clients lsh -E $TEI_EPS -b 2048 "Hello, world"
  
  # -E/--endpoints can be placed either before or after subcommand
  tei_clients -E $TEI_EPS info
"""

import argparse
import httpx
import json

from dataclasses import dataclass
from tclogger import logger
from typing import Union

from .tei_client import TEIClient, InfoResponse, TIMEOUT


# ============================================================================
# Data Classes
# ============================================================================


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


# ============================================================================
# Multi-Machine Client
# ============================================================================


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

    def _distribute_inputs(
        self,
        inputs: list[str],
        healthy: list[tuple[int, TEIClient, MachineInfo]],
    ) -> list[tuple[TEIClient, list[str], int, int]]:
        """Distribute inputs across healthy machines based on their weights.

        Args:
            inputs: List of input texts
            healthy: List of (index, client, machine_info) tuples

        Returns:
            List of (client, chunk, start_idx, end_idx) tuples
        """
        if not healthy:
            return []

        n_inputs = len(inputs)

        # Calculate weights (number of healthy GPU instances per machine)
        total_weight = sum(m.weight for _, _, m in healthy)
        if total_weight == 0:
            # Fallback to equal distribution
            total_weight = len(healthy)
            weights = [1] * len(healthy)
        else:
            weights = [m.weight for _, _, m in healthy]

        # Distribute inputs proportionally to weights
        distributions = []
        start_idx = 0

        for i, (_, client, machine) in enumerate(healthy):
            if i == len(healthy) - 1:
                # Last machine gets remaining inputs
                chunk = inputs[start_idx:]
                end_idx = n_inputs
            else:
                # Calculate chunk size based on weight
                chunk_size = max(1, int(n_inputs * weights[i] / total_weight))
                end_idx = min(start_idx + chunk_size, n_inputs)
                chunk = inputs[start_idx:end_idx]

            if chunk:
                distributions.append((client, chunk, start_idx, end_idx))
                start_idx = end_idx

            if start_idx >= n_inputs:
                break

        return distributions

    def embed(
        self,
        inputs: Union[str, list[str]],
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for input texts using multiple machines.

        Distributes inputs across healthy machines proportionally to their
        number of healthy GPU instances.

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

        # Distribute inputs across machines
        distributions = self._distribute_inputs(inputs, healthy)

        # Execute requests (synchronously for now, could be parallelized with threads)
        results: list[tuple[int, list[list[float]]]] = []
        errors = []

        for client, chunk, start_idx, end_idx in distributions:
            try:
                embs = client.embed(chunk, normalize=normalize, truncate=truncate)
                results.append((start_idx, embs))
            except Exception as e:
                errors.append((client.endpoint, e))
                # Mark machine as unhealthy
                for m in self.machines:
                    if m.endpoint == client.endpoint:
                        m.healthy = False
                        break

        if not results:
            raise ValueError(f"All requests failed: {errors}")

        # Sort by start index and combine results
        results.sort(key=lambda x: x[0])
        all_embeddings = []
        for _, embs in results:
            all_embeddings.extend(embs)

        self._log_okay(
            "embed",
            f"n={len(all_embeddings)}, machines={len(distributions)}",
        )

        return all_embeddings

    def lsh(
        self,
        inputs: Union[str, list[str]],
        bitn: int = 2048,
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[str]:
        """Generate LSH hash hex strings for input texts using multiple machines.

        Distributes inputs across healthy machines proportionally to their
        number of healthy GPU instances.

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

        # For small inputs or single machine, use simple path
        if len(inputs) <= 10 or len(healthy) == 1:
            _, client, _ = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return client.lsh(inputs, bitn=bitn, normalize=normalize, truncate=truncate)

        # Distribute inputs across machines
        distributions = self._distribute_inputs(inputs, healthy)

        # Execute requests
        results: list[tuple[int, list[str]]] = []
        errors = []

        for client, chunk, start_idx, end_idx in distributions:
            try:
                hashes = client.lsh(
                    chunk, bitn=bitn, normalize=normalize, truncate=truncate
                )
                results.append((start_idx, hashes))
            except Exception as e:
                errors.append((client.endpoint, e))
                for m in self.machines:
                    if m.endpoint == client.endpoint:
                        m.healthy = False
                        break

        if not results:
            raise ValueError(f"All requests failed: {errors}")

        # Sort and combine
        results.sort(key=lambda x: x[0])
        all_hashes = []
        for _, hashes in results:
            all_hashes.extend(hashes)

        self._log_okay(
            "lsh", f"n={len(all_hashes)}, bitn={bitn}, machines={len(distributions)}"
        )

        return all_hashes

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


# ============================================================================
# CLI
# ============================================================================


class TEIClientsArgParser:
    """Argument parser for TEI Clients CLI."""

    def __init__(self):
        # Create parent parser for common arguments
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser.add_argument(
            "-E",
            "--endpoints",
            type=str,
            required=True,
            help="Comma-separated list of tei_machine endpoints (required)",
        )
        parent_parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

        # Create main parser
        self.parser = argparse.ArgumentParser(
            description="TEI Clients - Connect to multiple TEI machines",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )

        # Add common arguments to main parser
        self.parser.add_argument(
            "-E",
            "--endpoints",
            type=str,
            required=False,  # Optional at root level
            help="Comma-separated list of tei_machine endpoints",
        )
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

        self._setup_subcommands(parent_parser)
        self.args = self.parser.parse_args()

    def _setup_subcommands(self, parent_parser):
        """Setup subcommands with parent parser."""
        # Action subcommands (all inherit from parent_parser)
        subparsers = self.parser.add_subparsers(dest="action", help="Action to perform")

        # health
        subparsers.add_parser(
            "health",
            help="Check health of all machines",
            parents=[parent_parser],
        )

        # info
        subparsers.add_parser(
            "info",
            help="Get info from all machines",
            parents=[parent_parser],
        )

        # embed
        embed_parser = subparsers.add_parser(
            "embed",
            help="Generate embeddings",
            parents=[parent_parser],
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
            parents=[parent_parser],
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
        health = self.clients.health()
        status_color = "okay" if health.status == "healthy" else "warn"
        getattr(logger, status_color)(f"Status: {health.status}")
        logger.mesg(f"Machines: {health.healthy_machines}/{health.total_machines}")
        logger.mesg(f"Instances: {health.healthy_instances}/{health.total_instances}")

    def run_info(self) -> None:
        """Get and display info from all machines."""
        infos = self.clients.info()
        if not infos:
            logger.warn("× No machine info available")
            return

        for i, info in enumerate(infos):
            logger.okay(f"[Machine {i+1}] {self.clients.endpoints[i]}")
            self._display_single_info(info)
            print()

    def _display_single_info(self, info: InfoResponse) -> None:
        """Display info for a single machine."""
        logger.mesg(f"Port: {info.port}")
        logger.mesg(f"Instances ({len(info.instances)}):")

        dash_len = 85
        logger.note("=" * dash_len)
        logger.note(f"{'GPU':<6} {'NAME':<40} {'ENDPOINT':<25} {'STATUS':<8}")
        logger.note("-" * dash_len)

        for inst in info.instances:
            gpu_str = str(inst.gpu_id) if inst.gpu_id is not None else "?"
            status_str = "healthy" if inst.healthy else "unhealthy"
            logger.mesg(
                f"{gpu_str:<6} {inst.name:<40} {inst.endpoint:<25} {status_str:<8}"
            )

        logger.note("=" * dash_len)

        logger.mesg(f"\nStats:")
        logger.mesg(f"  Total requests : {info.stats.total_requests}")
        logger.mesg(f"  Total inputs   : {info.stats.total_inputs}")
        logger.mesg(f"  Total errors   : {info.stats.total_errors}")

        if info.stats.requests_per_instance:
            logger.mesg(f"  Requests per instance:")
            for name, count in info.stats.requests_per_instance.items():
                logger.mesg(f"    {name}: {count}")

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
