"""TEI (Text Embeddings Inference) Client

This module provides a client for connecting to TEI services,
either the load-balanced tei_machine or individual tei_compose containers.
"""

# ANCHOR[id=client-clis]
CLI_EPILOG = """
Examples:
  # Connect to tei_machine (default port 28800)
  tei_client health                        # Check health
  tei_client info                          # Get server info
  tei_client embed "Hello, world"          # Embed single text
  tei_client embed "Hello" "World"         # Embed multiple texts
  tei_client lsh "Hello, world"            # Get LSH hash
  
  # Connect to specific endpoint
  tei_client -e "http://localhost:28800" health
  tei_client -e "http://localhost:28880" embed "Hello"  # Direct TEI container
  
  # With custom port (shorthand for localhost)
  tei_client -p 28800 health
  tei_client -p 28880 embed "Hello"
  
  # LSH with custom bit count
  tei_client lsh -b 2048 "Hello, world"
"""

import argparse
import httpx
import json
import orjson

from dataclasses import dataclass
from tclogger import logger, logstr, rows_to_table_str, dict_to_lines
from typing import Optional, Union

PORT = 28800  # default port for tei_machine
HOST = "localhost"


@dataclass
class HealthResponse:
    """Health check response."""

    status: str
    healthy: int
    total: int

    @classmethod
    def from_dict(cls, data: dict) -> "HealthResponse":
        return cls(
            status=data.get("status", "unknown"),
            healthy=data.get("healthy", 0),
            total=data.get("total", 0),
        )


@dataclass
class InstanceInfo:
    """Information about a single TEI instance."""

    name: str
    endpoint: str
    gpu_id: Optional[int]
    healthy: bool

    @classmethod
    def from_dict(cls, data: dict) -> "InstanceInfo":
        return cls(
            name=data.get("name", ""),
            endpoint=data.get("endpoint", ""),
            gpu_id=data.get("gpu_id"),
            healthy=data.get("healthy", False),
        )


@dataclass
class MachineStats:
    """Statistics for the machine."""

    total_requests: int
    total_inputs: int
    total_errors: int
    requests_per_instance: dict[str, int]

    @classmethod
    def from_dict(cls, data: dict) -> "MachineStats":
        return cls(
            total_requests=data.get("total_requests", 0),
            total_inputs=data.get("total_inputs", 0),
            total_errors=data.get("total_errors", 0),
            requests_per_instance=data.get("requests_per_instance", {}),
        )


@dataclass
class InfoResponse:
    """Info response from tei_machine."""

    port: int
    instances: list[InstanceInfo]
    stats: MachineStats

    @classmethod
    def from_dict(cls, data: dict) -> "InfoResponse":
        return cls(
            port=data.get("port", 0),
            instances=[InstanceInfo.from_dict(i) for i in data.get("instances", [])],
            stats=MachineStats.from_dict(data.get("stats", {})),
        )


class TEIClient:
    """Synchronous client for TEI services.

    Can connect to either:
    - tei_machine (load-balanced proxy, default port 28800)
    - tei_compose containers (direct TEI, ports 28880+)

    Example:
        client = TEIClient("http://localhost:28800")
        embs = client.embed(["Hello", "World"])
        lsh_hashes = client.lsh(["Hello", "World"])
    """

    def __init__(
        self,
        endpoint: str = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
    ):
        """Initialize TEI client.

        Args:
            endpoint: Full endpoint URL (e.g., "http://localhost:28800").
                     If provided, host and port are ignored.
            host: Server host (default: localhost)
            port: Server port (default: 28800)
            verbose: Enable verbose logging
        """
        if endpoint:
            self.endpoint = endpoint.rstrip("/")
        else:
            self.endpoint = f"http://{host}:{port}"

        self.verbose = verbose
        self.client = httpx.Client(timeout=httpx.Timeout(60.0))

    def close(self) -> None:
        """Close the HTTP client."""
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self) -> "TEIClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _log_fail(self, action: str, error: Exception) -> None:
        """Log error message."""
        if self.verbose:
            logger.warn(f"× TEI {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        """Log success message."""
        if self.verbose:
            logger.okay(f"✓ TEI {action}: {message}")

    def _extract_error_detail(self, e: httpx.HTTPStatusError) -> str:
        """Extract error detail from HTTP status error response."""
        try:
            return e.response.json().get("detail", str(e))
        except Exception:
            return str(e)

    def health(self) -> HealthResponse:
        """Check health status of the TEI service.

        Returns:
            HealthResponse with status, healthy count, and total count.

        Raises:
            httpx.HTTPError: On connection or request error
        """
        try:
            resp = self.client.get(f"{self.endpoint}/health")
            resp.raise_for_status()
            data = resp.json()
            result = HealthResponse.from_dict(data)
            self._log_okay("health", f"status={result.status}")
            return result
        except httpx.HTTPStatusError as e:
            # tei_machine returns 503 with health details when unhealthy
            try:
                data = e.response.json()
                if "detail" in data and isinstance(data["detail"], dict):
                    return HealthResponse.from_dict(data["detail"])
            except Exception:
                pass
            self._log_fail("health", e)
            raise
        except Exception as e:
            self._log_fail("health", e)
            raise

    def is_healthy(self) -> bool:
        """Quick health check returning boolean.

        Returns:
            True if service is healthy, False otherwise.
        """
        try:
            result = self.health()
            return result.status == "healthy" or result.healthy > 0
        except Exception:
            return False

    def log_machine_health(self) -> None:
        health = self.health()
        logger.mesg(f"* Healthy: {logstr.okay(health.healthy)}/{health.total}")

    def info(self) -> InfoResponse:
        """Get detailed information about tei_machine.

        Note: This endpoint is only available on tei_machine, not on
        individual TEI containers.

        Returns:
            InfoResponse with port, instances, and stats.

        Raises:
            httpx.HTTPError: On connection or request error
        """
        try:
            resp = self.client.get(f"{self.endpoint}/info")
            resp.raise_for_status()
            data = resp.json()
            result = InfoResponse.from_dict(data)
            self._log_okay(
                "info", f"port={result.port}, instances={len(result.instances)}"
            )
            return result
        except Exception as e:
            self._log_fail("info", e)
            raise

    def log_machine_info(self) -> None:
        """Log info for a single machine."""
        info = self.info()
        machine_dict = {
            "port": info.port,
            "instances": len(info.instances),
        }
        machine_str = dict_to_lines(machine_dict, key_prefix="* ")
        logger.note(f"Machine Info:")
        print(machine_str)

        rows: list[list] = []
        for inst in info.instances:
            row = [
                str(inst.gpu_id) if inst.gpu_id is not None else "?",
                inst.name,
                inst.endpoint,
                logstr.okay("healthy") if inst.healthy else logstr.fail("sick"),
            ]
            rows.append(row)
        table_str = rows_to_table_str(
            rows=rows, headers=["gpu", "name", "endpoint", "status"]
        )
        print(table_str)

        logger.note(f"Stats:")
        stats_dict = {
            "total_requests": info.stats.total_requests,
            "total_inputs": info.stats.total_inputs,
            "total_errors": info.stats.total_errors,
        }
        stats_tr = dict_to_lines(stats_dict, key_prefix="* ")
        print(stats_tr)

        if info.stats.requests_per_instance:
            logger.note(f"Requests per instance:")
            reqs_dict = {}
            for name, count in info.stats.requests_per_instance.items():
                reqs_dict[name] = count
            reqs_tr = dict_to_lines(reqs_dict, key_prefix="* ")
            print(reqs_tr)

    def embed(
        self,
        inputs: Union[str, list[str]],
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for input texts.

        Args:
            inputs: Single text or list of texts to embed.
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            List of embedding vectors (list of floats).

        Raises:
            httpx.HTTPError: On connection or request error
            ValueError: On server-side errors
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        payload = {
            "inputs": inputs,
            "normalize": normalize,
            "truncate": truncate,
        }

        try:
            resp = self.client.post(f"{self.endpoint}/embed", json=payload)
            resp.raise_for_status()
            embs = resp.json()
            self._log_okay(
                "embed",
                f"n={len(embs)}, dims={len(embs[0]) if embs else 0}",
            )
            return embs
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e)
            self._log_fail("embed", error_detail)
            raise ValueError(f"Embed failed: {error_detail}") from e
        except Exception as e:
            self._log_fail("embed", e)
            raise

    def lsh(
        self,
        inputs: Union[str, list[str]],
        bitn: int = 2048,
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[str]:
        """Generate LSH hash hex strings for input texts.

        Note: This endpoint is only available on tei_machine, not on
        individual TEI containers.

        Args:
            inputs: Single text or list of texts.
            bitn: Number of LSH hash bits (default: 2048, range: 64-8192)
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            List of hex strings representing LSH hashes.

        Raises:
            httpx.HTTPError: On connection or request error
            ValueError: On server-side errors
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        payload = {
            "inputs": inputs,
            "bitn": bitn,
            "normalize": normalize,
            "truncate": truncate,
        }

        try:
            resp = self.client.post(f"{self.endpoint}/lsh", json=payload)
            resp.raise_for_status()
            hashes = resp.json()
            self._log_okay("lsh", f"n={len(hashes)}, bitn={bitn}")
            return hashes
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e)
            self._log_fail("lsh", error_detail)
            raise ValueError(f"LSH failed: {error_detail}") from e
        except Exception as e:
            self._log_fail("lsh", e)
            raise


class AsyncTEIClient:
    """Asynchronous client for TEI services.

    Designed for high-throughput scenarios where multiple concurrent requests
    are needed. Uses httpx.AsyncClient to avoid thread pool overhead.

    Example:
        async with AsyncTEIClient("http://localhost:28800") as client:
            embs = await client.embed(["Hello", "World"])
            hashes = await client.lsh(["Hello", "World"])
    """

    def __init__(
        self,
        endpoint: str = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
    ):
        """Initialize async TEI client.

        Args:
            endpoint: Full endpoint URL (e.g., "http://localhost:28800").
                     If provided, host and port are ignored.
            host: Server host (default: localhost)
            port: Server port (default: 28800)
            verbose: Enable verbose logging
        """
        if endpoint:
            self.endpoint = endpoint.rstrip("/")
        else:
            self.endpoint = f"http://{host}:{port}"

        self.verbose = verbose
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncTEIClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def _log_fail(self, action: str, error: Exception) -> None:
        """Log error message."""
        if self.verbose:
            logger.warn(f"× AsyncTEI {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        """Log success message."""
        if self.verbose:
            logger.okay(f"✓ AsyncTEI {action}: {message}")

    def _extract_error_detail(self, e: httpx.HTTPStatusError) -> str:
        """Extract error detail from HTTP status error response."""
        try:
            return e.response.json().get("detail", str(e))
        except Exception:
            return str(e)

    async def health(self) -> HealthResponse:
        """Check health status of the TEI service.

        Returns:
            HealthResponse with status, healthy count, and total count.
        """
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.endpoint}/health")
            resp.raise_for_status()
            data = resp.json()
            result = HealthResponse.from_dict(data)
            self._log_okay("health", f"status={result.status}")
            return result
        except httpx.HTTPStatusError as e:
            try:
                data = e.response.json()
                if "detail" in data and isinstance(data["detail"], dict):
                    return HealthResponse.from_dict(data["detail"])
            except Exception:
                pass
            self._log_fail("health", e)
            raise
        except Exception as e:
            self._log_fail("health", e)
            raise

    async def embed(
        self,
        inputs: Union[str, list[str]],
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for input texts.

        Args:
            inputs: Single text or list of texts to embed.
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            List of embedding vectors (list of floats).
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        payload = {
            "inputs": inputs,
            "normalize": normalize,
            "truncate": truncate,
        }

        client = await self._get_client()
        try:
            # Use orjson for fast pre-serialization to avoid blocking
            content = orjson.dumps(payload)
            resp = await client.post(
                f"{self.endpoint}/embed",
                content=content,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            embs = resp.json()
            self._log_okay(
                "embed",
                f"n={len(embs)}, dims={len(embs[0]) if embs else 0}",
            )
            return embs
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e)
            self._log_fail("embed", error_detail)
            raise ValueError(f"Embed failed: {error_detail}") from e
        except Exception as e:
            self._log_fail("embed", e)
            raise

    async def lsh(
        self,
        inputs: Union[str, list[str]],
        bitn: int = 2048,
        normalize: bool = True,
        truncate: bool = True,
    ) -> list[str]:
        """Generate LSH hash hex strings for input texts.

        Args:
            inputs: Single text or list of texts.
            bitn: Number of LSH hash bits (default: 2048, range: 64-8192)
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            List of hex strings representing LSH hashes.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        payload = {
            "inputs": inputs,
            "bitn": bitn,
            "normalize": normalize,
            "truncate": truncate,
        }

        client = await self._get_client()
        try:
            # Use orjson for fast pre-serialization to avoid blocking
            content = orjson.dumps(payload)
            resp = await client.post(
                f"{self.endpoint}/lsh",
                content=content,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            hashes = resp.json()
            self._log_okay("lsh", f"n={len(hashes)}, bitn={bitn}")
            return hashes
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e)
            self._log_fail("lsh", error_detail)
            raise ValueError(f"LSH failed: {error_detail}") from e
        except Exception as e:
            self._log_fail("lsh", e)
            raise


class TEIClientArgParser:
    """Argument parser for TEI Client CLI."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="TEI Client - Connect to TEI services",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        """Setup all command-line arguments."""
        # Connection options
        self.parser.add_argument(
            "-e",
            "--endpoint",
            type=str,
            default=None,
            help="Full endpoint URL (e.g., http://localhost:28800)",
        )
        self.parser.add_argument(
            "-p",
            "--port",
            type=int,
            default=PORT,
            help=f"Server port (default: {PORT})",
        )
        self.parser.add_argument(
            "-H",
            "--host",
            type=str,
            default=HOST,
            help=f"Server host (default: {HOST})",
        )
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

        # Action subcommands
        subparsers = self.parser.add_subparsers(dest="action", help="Action to perform")

        # health
        subparsers.add_parser("health", help="Check service health")

        # info
        subparsers.add_parser("info", help="Get service info (tei_machine only)")

        # embed
        embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
        embed_parser.add_argument(
            "texts",
            nargs="+",
            help="Texts to embed",
        )

        # lsh
        lsh_parser = subparsers.add_parser(
            "lsh", help="Generate LSH hashes (tei_machine only)"
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


class TEIClientCLI:
    """CLI interface for TEI Client operations."""

    def __init__(self, client: TEIClient):
        """Initialize CLI with a TEI client.

        Args:
            client: TEIClient instance to use for operations
        """
        self.client = client

    def run_health(self) -> None:
        """Run health check and display results."""
        self.client.log_machine_health()

    def run_info(self) -> None:
        """Get and display server info."""
        self.client.log_machine_info()

    def run_embed(self, texts: list[str]) -> None:
        """Generate and display embeddings.

        Args:
            texts: List of texts to embed
        """
        if not texts:
            logger.warn("× No input texts provided")
            return

        embs = self.client.embed(texts)
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

        hashes = self.client.lsh(texts, bitn=bitn)
        for text, hash_str in zip(texts, hashes):
            text_preview = text[:40] + "..." if len(text) > 40 else text
            hash_preview = hash_str[:32] + "..." if len(hash_str) > 32 else hash_str
            logger.mesg(f"'{text_preview}'")
            logger.file(f"  → {hash_preview}")


def main():
    """Main entry point for CLI."""
    arg_parser = TEIClientArgParser()
    args = arg_parser.args

    if args.action is None:
        arg_parser.parser.print_help()
        return

    client = TEIClient(
        endpoint=args.endpoint,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

    try:
        cli = TEIClientCLI(client)

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
        logger.hint(f"  Is the TEI service running at {client.endpoint}?")
    except Exception as e:
        logger.warn(f"× Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_client.py#client-clis
