"""TEI Multi-Machine Client - Production Version

Distribute embed/lsh requests across multiple TEI machines with async pipeline.
For verbose logging and stats, use TEIClientsWithStats.
"""

from .clients_core import _TEIClientsBase, _TEIClientsPipeline


# ANCHOR[id=clients-clis]
CLI_EPILOG = """
Examples:
    export TEI_EPS="$TEI_MACHINE_A_URL,$TEI_MACHINE_B_URL"
  
    # Unified CLI replacement
    tei client health -E "$TEI_EPS"
    tei client info -E "$TEI_EPS"
    tei client embed -E "$TEI_EPS" "Hello" "World"
    tei client lsh -E "$TEI_EPS" "Hello"
    tei client lsh -E "$TEI_EPS" -b 2048 "Hello, world"
    tei client rerank -E "$TEI_EPS" -q "hello" -p "hi" "good" "你好"
"""


class TEIClients(_TEIClientsBase):
    """Production multi-machine TEI client with async pipeline scheduling."""

    def __init__(self, endpoints: list[str]):
        """Initialize multi-machine TEI client.

        Args:
            endpoints: List of tei_machine endpoint URLs
        """
        # Set verbose before calling super().__init__()
        self._verbose = False
        super().__init__(endpoints)

        # Create pipeline with no callbacks (production mode)
        self._pipeline = _TEIClientsPipeline(
            machine_scheduler=self.machine_scheduler,
            on_progress=None,
            on_complete=None,
        )


def main():
    """Main entry point for CLI."""
    from .clients_cli import (
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
