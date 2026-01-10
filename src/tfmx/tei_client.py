"""TEI (Text Embeddings Inference) Client

This module provides a client for connecting to TEI services,
either the load-balanced tei_machine or individual tei_compose containers.
"""

# ANCHOR[id=client-clis]
CLI_EPILOG = """
Examples:
  # Connect to tei_machine (default port 28800)
  tei_client health                         # Check health
  tei_client info                           # Get server info
  tei_client embed "Hello, world!"          # Embed single text
  tei_client embed "Hello" "World"          # Embed multiple texts
  tei_client lsh "Hello, world!"            # Get LSH hash
  
  # Connect to specific endpoint
  tei_client -e "http://localhost:28800" health
  tei_client -e "http://localhost:28880" embed "Hello"  # Direct TEI container
  
  # With custom port (shorthand for localhost)
  tei_client -p 28800 health
  tei_client -p 28880 embed "Hello"
  
  # LSH with custom bit count
  tei_client lsh -b 1024 "Hello, world!"
  tei_client lsh --bitn 4096 "Hello"
  
  # Batch embed from stdin (one text per line)
  cat texts.txt | tei_client embed -
  echo -e "Hello\\nWorld" | tei_client embed -
  
  # Output format options
  tei_client embed "Hello" -o json          # JSON output (default)
  tei_client embed "Hello" -o dims          # Show dimensions only
  tei_client embed "Hello" -o preview       # Preview first few values
"""

import argparse
import sys

from dataclasses import dataclass
from typing import Optional, Union

import httpx
import numpy as np

from tclogger import logger


PORT = 28800  # tei_machine default port
HOST = "localhost"
TIMEOUT = 60.0


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


# Synchronous Client


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
        timeout: float = TIMEOUT,
        verbose: bool = False,
    ):
        """Initialize TEI client.

        Args:
            endpoint: Full endpoint URL (e.g., "http://localhost:28800").
                     If provided, host and port are ignored.
            host: Server host (default: localhost)
            port: Server port (default: 28800)
            timeout: Request timeout in seconds (default: 60.0)
            verbose: Enable verbose logging
        """
        if endpoint:
            self.endpoint = endpoint.rstrip("/")
        else:
            self.endpoint = f"http://{host}:{port}"

        self.timeout = timeout
        self.verbose = verbose
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout))
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "TEIClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _log_error(self, action: str, error: Exception) -> None:
        """Log error message."""
        if self.verbose:
            logger.warn(f"× TEI {action} error: {error}")

    def _log_success(self, action: str, message: str) -> None:
        """Log success message."""
        if self.verbose:
            logger.okay(f"✓ TEI {action}: {message}")

    def health(self) -> HealthResponse:
        """Check health status of the TEI service.

        Returns:
            HealthResponse with status, healthy count, and total count.

        Raises:
            httpx.HTTPError: On connection or request error
        """
        client = self._get_client()
        try:
            resp = client.get(f"{self.endpoint}/health")
            resp.raise_for_status()
            data = resp.json()
            result = HealthResponse.from_dict(data)
            self._log_success("health", f"status={result.status}")
            return result
        except httpx.HTTPStatusError as e:
            # Try to parse error response (tei_machine returns 503 when unhealthy)
            try:
                data = e.response.json()
                if "detail" in data and isinstance(data["detail"], dict):
                    return HealthResponse.from_dict(data["detail"])
            except Exception:
                pass
            self._log_error("health", e)
            raise
        except Exception as e:
            self._log_error("health", e)
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

    def info(self) -> InfoResponse:
        """Get detailed information about tei_machine.

        Note: This endpoint is only available on tei_machine, not on
        individual TEI containers.

        Returns:
            InfoResponse with port, instances, and stats.

        Raises:
            httpx.HTTPError: On connection or request error
        """
        client = self._get_client()
        try:
            resp = client.get(f"{self.endpoint}/info")
            resp.raise_for_status()
            data = resp.json()
            result = InfoResponse.from_dict(data)
            self._log_success(
                "info", f"port={result.port}, instances={len(result.instances)}"
            )
            return result
        except Exception as e:
            self._log_error("info", e)
            raise

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
        client = self._get_client()

        # Ensure inputs is a list
        if isinstance(inputs, str):
            inputs = [inputs]

        payload = {
            "inputs": inputs,
            "normalize": normalize,
            "truncate": truncate,
        }

        try:
            resp = client.post(f"{self.endpoint}/embed", json=payload)
            resp.raise_for_status()
            embs = resp.json()
            self._log_success(
                "embed",
                f"n={len(embs)}, dims={len(embs[0]) if embs else 0}",
            )
            return embs
        except httpx.HTTPStatusError as e:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            self._log_error("embed", error_detail)
            raise ValueError(f"Embed failed: {error_detail}") from e
        except Exception as e:
            self._log_error("embed", e)
            raise

    def embed_numpy(
        self,
        inputs: Union[str, list[str]],
        normalize: bool = True,
        truncate: bool = True,
    ) -> np.ndarray:
        """Generate embeddings as numpy array.

        Args:
            inputs: Single text or list of texts to embed.
            normalize: Whether to normalize embeddings (default: True)
            truncate: Whether to truncate long inputs (default: True)

        Returns:
            Numpy array of shape (n, dims) containing embeddings.
        """
        embs = self.embed(inputs, normalize, truncate)
        return np.array(embs, dtype=np.float32)

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
        client = self._get_client()

        # Ensure inputs is a list
        if isinstance(inputs, str):
            inputs = [inputs]

        payload = {
            "inputs": inputs,
            "bitn": bitn,
            "normalize": normalize,
            "truncate": truncate,
        }

        try:
            resp = client.post(f"{self.endpoint}/lsh", json=payload)
            resp.raise_for_status()
            hashes = resp.json()
            self._log_success("lsh", f"n={len(hashes)}, bitn={bitn}")
            return hashes
        except httpx.HTTPStatusError as e:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            self._log_error("lsh", error_detail)
            raise ValueError(f"LSH failed: {error_detail}") from e
        except Exception as e:
            self._log_error("lsh", e)
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
            "-t",
            "--timeout",
            type=float,
            default=TIMEOUT,
            help=f"Request timeout in seconds (default: {TIMEOUT})",
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
            nargs="*",
            help="Texts to embed (use '-' to read from stdin)",
        )
        embed_parser.add_argument(
            "-o",
            "--output",
            type=str,
            choices=["json", "dims", "preview"],
            default="json",
            help="Output format: json (full), dims (dimensions only), preview (first values)",
        )
        embed_parser.add_argument(
            "--no-normalize",
            action="store_true",
            help="Disable normalization",
        )
        embed_parser.add_argument(
            "--no-truncate",
            action="store_true",
            help="Disable truncation",
        )

        # lsh
        lsh_parser = subparsers.add_parser(
            "lsh", help="Generate LSH hashes (tei_machine only)"
        )
        lsh_parser.add_argument(
            "texts",
            nargs="*",
            help="Texts to hash (use '-' to read from stdin)",
        )
        lsh_parser.add_argument(
            "-b",
            "--bitn",
            type=int,
            default=2048,
            help="Number of LSH bits (default: 2048)",
        )
        lsh_parser.add_argument(
            "--no-normalize",
            action="store_true",
            help="Disable normalization",
        )
        lsh_parser.add_argument(
            "--no-truncate",
            action="store_true",
            help="Disable truncation",
        )


def read_inputs_from_args(texts: list[str]) -> list[str]:
    """Read input texts from args or stdin.

    Args:
        texts: List of texts from command line, may contain '-' for stdin

    Returns:
        List of input texts
    """
    if not texts or texts == ["-"]:
        # Read from stdin
        inputs = []
        for line in sys.stdin:
            line = line.strip()
            if line:
                inputs.append(line)
        return inputs
    return texts


def format_health_output(health: HealthResponse) -> None:
    """Format and print health response."""
    status_color = "okay" if health.status == "healthy" else "warn"
    getattr(logger, status_color)(f"Status: {health.status}")
    logger.mesg(f"Healthy: {health.healthy}/{health.total}")


def format_info_output(info: InfoResponse) -> None:
    """Format and print info response."""
    logger.mesg(f"Port: {info.port}")
    logger.mesg(f"Instances ({len(info.instances)}):")

    dash_len = 75
    logger.note("=" * dash_len)
    logger.note(f"{'GPU':<6} {'NAME':<35} {'ENDPOINT':<22} {'STATUS':<8}")
    logger.note("-" * dash_len)

    for inst in info.instances:
        gpu_str = str(inst.gpu_id) if inst.gpu_id is not None else "?"
        status_str = "healthy" if inst.healthy else "unhealthy"
        logger.mesg(f"{gpu_str:<6} {inst.name:<35} {inst.endpoint:<22} {status_str:<8}")

    logger.note("=" * dash_len)

    # Stats
    logger.mesg(f"\nStats:")
    logger.mesg(f"  Total requests : {info.stats.total_requests}")
    logger.mesg(f"  Total inputs   : {info.stats.total_inputs}")
    logger.mesg(f"  Total errors   : {info.stats.total_errors}")

    if info.stats.requests_per_instance:
        logger.mesg(f"  Requests per instance:")
        for name, count in info.stats.requests_per_instance.items():
            logger.mesg(f"    {name}: {count}")


def format_embed_output(
    embs: list[list[float]], output_format: str, texts: list[str]
) -> None:
    """Format and print embeddings."""
    import json

    if output_format == "json":
        print(json.dumps(embs, indent=2))
    elif output_format == "dims":
        n = len(embs)
        dims = len(embs[0]) if n > 0 else 0
        logger.okay(f"Embeddings: n={n}, dims={dims}")
    elif output_format == "preview":
        for i, (text, emb) in enumerate(zip(texts, embs)):
            preview = emb[:5] if len(emb) > 5 else emb
            preview_str = ", ".join(f"{v:.4f}" for v in preview)
            text_preview = text[:30] + "..." if len(text) > 30 else text
            logger.mesg(f"[{i}] '{text_preview}': [{preview_str}, ...]")
        dims = len(embs[0]) if embs else 0
        logger.note(f"Total: {len(embs)} embeddings, {dims} dimensions")


def format_lsh_output(hashes: list[str], texts: list[str]) -> None:
    """Format and print LSH hashes."""
    for text, hash_str in zip(texts, hashes):
        text_preview = text[:40] + "..." if len(text) > 40 else text
        # Show first 32 chars of hash for readability
        hash_preview = hash_str[:32] + "..." if len(hash_str) > 32 else hash_str
        logger.mesg(f"'{text_preview}'")
        logger.file(f"  → {hash_preview}")


def main():
    """Main entry point for CLI."""
    arg_parser = TEIClientArgParser()
    args = arg_parser.args

    # Show help if no action specified
    if args.action is None:
        arg_parser.parser.print_help()
        return

    # Create client
    client = TEIClient(
        endpoint=args.endpoint,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    try:
        if args.action == "health":
            health = client.health()
            format_health_output(health)

        elif args.action == "info":
            info = client.info()
            format_info_output(info)

        elif args.action == "embed":
            texts = read_inputs_from_args(args.texts)
            if not texts:
                logger.warn("× No input texts provided")
                return

            normalize = not args.no_normalize
            truncate = not args.no_truncate

            embs = client.embed(texts, normalize=normalize, truncate=truncate)
            format_embed_output(embs, args.output, texts)

        elif args.action == "lsh":
            texts = read_inputs_from_args(args.texts)
            if not texts:
                logger.warn("× No input texts provided")
                return

            normalize = not args.no_normalize
            truncate = not args.no_truncate

            hashes = client.lsh(
                texts, bitn=args.bitn, normalize=normalize, truncate=truncate
            )
            format_lsh_output(hashes, texts)

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
