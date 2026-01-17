"""TEI Multi-Machine Client - Production Version

High-performance client for distributing embed/lsh requests across
multiple TEI machines using optimized async pipeline scheduling.

This is the clean production version without verbose logging.
For testing/benchmarking with detailed stats, use TEIClientsWithStats.

Key Features:
- Async pipeline scheduling for maximum throughput
- Auto-loads optimal batch sizes from tei_clients.config.json
- Smart routing: round-robin for small batches, pipeline for large batches
- Automatic health checking and recovery
- Iterator support for memory-efficient processing

Usage:
    from tfmx import TEIClients
    
    endpoints = ["http://machine1:28800", "http://machine2:28800"]
    with TEIClients(endpoints) as clients:
        embeddings = clients.embed(texts)
        lsh_hashes = clients.lsh(texts, bitn=2048)

See TEI.md for detailed usage guide.
"""

import argparse
import asyncio
import httpx
import json
from typing import Union, Iterable

from .tei_client import TEIClient, AsyncTEIClient, InfoResponse, TIMEOUT
from .tei_clients_core import (
    MachineState,
    MachineScheduler,
    ClientsHealthResponse,
    _TEIClientsPipeline,
)
from .tei_performance import ExplorationConfig


# ANCHOR[id=clients-clis]
CLI_EPILOG = """
Examples:
  export TEI_EPS="http://localhost:28800,http://ai122:28800"
  
  # Action comes first
  tei_clients health -E $TEI_EPS
  tei_clients info -E $TEI_EPS
  tei_clients embed -E $TEI_EPS "Hello" "World"
  tei_clients lsh -E $TEI_EPS "Hello"
  tei_clients lsh -E $TEI_EPS -b 2048 "Hello, world"
"""


class TEIClients:
    """Production multi-machine TEI client with optimal pipeline scheduling.

    Automatically loads optimal batch sizes from tei_clients.config.json.
    Uses round-robin for small batches (<1000), pipeline for large batches.

    Example:
        ```python
        clients = TEIClients([
            "http://machine1:28800",
            "http://machine2:28800",
        ])

        embeddings = clients.embed(texts)
        lsh_hashes = clients.lsh(texts, bitn=2048)
        clients.close()
        ```

    Context manager:
        ```python
        with TEIClients(endpoints) as clients:
            results = clients.lsh(large_dataset, bitn=2048)
        ```
    """

    def __init__(
        self,
        endpoints: list[str],
        timeout: float = TIMEOUT,
    ):
        """Initialize multi-machine TEI client.

        Args:
            endpoints: List of tei_machine endpoint URLs
                      (e.g., ["http://machine1:28800", "http://machine2:28800"])
            timeout: Request timeout in seconds (default: 60.0)
        """
        self.endpoints = [ep.rstrip("/") for ep in endpoints]
        self.timeout = timeout

        # Create underlying clients for each endpoint
        self.clients: list[TEIClient] = [
            TEIClient(endpoint=ep, timeout=timeout, verbose=False)
            for ep in self.endpoints
        ]

        # Create async clients for pipeline
        self.async_clients: list[AsyncTEIClient] = [
            AsyncTEIClient(endpoint=ep, timeout=timeout, verbose=False)
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

        # Pipeline executor (composition)
        self._pipeline = _TEIClientsPipeline(
            machine_scheduler=self.machine_scheduler,
            on_progress=None,  # No verbose logging in production
            on_complete=None,
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

    def close(self) -> None:
        """Close all HTTP clients."""
        for client in self.clients:
            client.close()

    async def aclose(self) -> None:
        """Close all async HTTP clients."""
        for async_client in self.async_clients:
            await async_client.close()

    def __enter__(self) -> "TEIClients":
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


def main():
    """Main entry point for CLI."""
    from .tei_clients_cli import (
        TEIClientsArgParserBase,
        run_cli_main,
    )

    run_cli_main(
        parser_class=TEIClientsArgParserBase,
        clients_class=TEIClients,
        description="TEI Clients - Connect to multiple TEI machines",
        epilog=CLI_EPILOG,
        extra_args=None,  # No extra args for production version
    )


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_clients.py#clients-clis
