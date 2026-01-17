"""TEI Multi-Machine Client with Stats - Testing/Exploration Version

Adds verbose logging and performance stats to TEIClients.
For production use without overhead, use TEIClients.
"""

from tclogger import logger, logstr

from .tei_clients_core import _TEIClientsBase, _TEIClientsPipeline


# ANCHOR[id=clients-stats-clis]
CLI_EPILOG = """
Examples:
  export TEI_EPS="http://localhost:28800,http://ai122:28800"
  
  # Action comes first
  tei_clients_stats health -E $TEI_EPS -v
  tei_clients_stats embed -E $TEI_EPS -v "Hello" "World"
  tei_clients_stats lsh -E $TEI_EPS -v "Hello, world"
"""


class TEIClientsWithStats(_TEIClientsBase):
    """Multi-machine TEI client with verbose logging and performance stats."""

    def __init__(self, endpoints: list[str], verbose: bool = False):
        """Initialize multi-machine TEI client with stats.

        Args:
            endpoints: List of tei_machine endpoint URLs
            verbose: Enable verbose logging and progress indicators
        """
        # Set verbose before calling super().__init__()
        self._verbose = verbose
        self.verbose = verbose
        super().__init__(endpoints)

        # Create pipeline with stats callbacks
        self._pipeline = _TEIClientsPipeline(
            machine_scheduler=self.machine_scheduler,
            on_progress=self._log_progress,
            on_complete=self._log_complete,
        )

    def _load_config(self) -> None:
        """Load optimal configurations from saved config file with verbose logging."""
        super()._load_config()

        # Add verbose logging
        if self.verbose:
            for machine in self.machines:
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


def main():
    """Main entry point for CLI."""
    from .tei_clients_cli import (
        TEIClientsArgParserBase,
        run_cli_main,
    )

    run_cli_main(
        parser_class=TEIClientsArgParserBase,
        clients_class=TEIClientsWithStats,
        description="TEI Clients with Stats - Multi-machine client with verbose logging",
        epilog=CLI_EPILOG,
        extra_args={"verbose": True},  # Add --verbose flag for stats version
    )


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_clients_stats.py#clients-stats-clis
