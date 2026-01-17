"""Shared CLI Infrastructure for TEI Clients

Provides base classes for argument parsing and CLI operations,
following DRY principle to avoid code duplication between
tei_clients.py and tei_clients_stats.py.
"""

import argparse
import json
import httpx
from typing import Protocol, Union
from tclogger import logger


class TEIClientsProtocol(Protocol):
    """Protocol for TEI clients (duck typing)."""

    endpoints: list[str]
    machines: list

    def embed(self, inputs, normalize=True, truncate=True) -> list[list[float]]: ...

    def lsh(self, inputs, bitn=2048) -> list[str]: ...

    def close(self) -> None: ...


class TEIClientsArgParserBase:
    """Base argument parser for TEI Clients CLI.

    Implements common argument parsing logic shared between
    tei_clients and tei_clients_stats.
    """

    def __init__(self, description: str, epilog: str, extra_common_args: dict = None):
        """Initialize base parser.

        Args:
            description: Parser description
            epilog: Examples/help epilog
            extra_common_args: Additional common arguments to add
                              (e.g., {'verbose': True} for stats version)
        """
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=epilog,
        )

        self.extra_common_args = extra_common_args or {}
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        """Setup all command-line arguments."""
        # Create subparsers FIRST (so action comes before options)
        subparsers = self.parser.add_subparsers(
            dest="action",
            help="Action to perform",
            required=True,
        )

        # Common parent parser for shared arguments
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser.add_argument(
            "-E",
            "--endpoints",
            type=str,
            required=False,
            help="Comma-separated list of tei_machine endpoints",
        )

        # Add extra common arguments (e.g., --verbose for stats version)
        if "verbose" in self.extra_common_args:
            parent_parser.add_argument(
                "-v",
                "--verbose",
                action="store_true",
                help="Enable verbose output and progress logging",
            )

        # health - check health of all machines
        subparsers.add_parser(
            "health",
            help="Check health of all machines",
            parents=[parent_parser],
        )

        # info - get info from all machines
        subparsers.add_parser(
            "info",
            help="Get info from all machines",
            parents=[parent_parser],
        )

        # embed - generate embeddings
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

        # lsh - generate LSH hashes
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


class TEIClientsCLIBase:
    """Base CLI interface for TEI Clients operations.

    Implements common CLI logic shared between tei_clients and tei_clients_stats.
    """

    def __init__(self, clients: TEIClientsProtocol):
        """Initialize CLI with TEI clients.

        Args:
            clients: TEI clients instance (TEIClients or TEIClientsWithStats)
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


def run_cli_main(
    parser_class: type,
    clients_class: type,
    description: str,
    epilog: str,
    extra_args: dict = None,
) -> None:
    """Shared main entry point for CLI.

    Args:
        parser_class: ArgParser class to use
        clients_class: TEI clients class to instantiate
        description: Parser description
        epilog: Parser epilog
        extra_args: Extra arguments for parser (e.g., {'verbose': True})
    """
    arg_parser = parser_class(description, epilog, extra_args)
    args = arg_parser.args

    # Validate endpoints argument
    if not args.endpoints:
        logger.warn("× Error: -E/--endpoints is required")
        arg_parser.parser.print_help()
        return

    endpoints = [ep.strip() for ep in args.endpoints.split(",")]

    # Create clients with appropriate arguments
    clients_kwargs = {"endpoints": endpoints}
    if hasattr(args, "verbose"):
        clients_kwargs["verbose"] = args.verbose

    clients = clients_class(**clients_kwargs)

    try:
        cli = TEIClientsCLIBase(clients)
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
