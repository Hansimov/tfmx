"""
TEI Server Management

This module provides management for TEI (Text Embeddings Inference) servers
using Docker Compose for multi-GPU deployment.
"""

import argparse
from typing import TypedDict, Optional

from tclogger import logger

from tfmx.tei_compose import TEIComposeManager, SERVER_PORT, MODEL_NAME


class TEIServerConfigsType(TypedDict):
    port: int
    model_name: str
    project_name: Optional[str]
    gpu_ids: Optional[str]
    hf_token: Optional[str]
    verbose: bool


class TEIServer:
    """TEI Server using Docker Compose for multi-GPU deployment."""

    def __init__(
        self,
        port: int = SERVER_PORT,
        model_name: str = None,
        project_name: str = None,
        hf_token: str = None,
        gpu_ids: str = None,
        verbose: bool = False,
    ):
        self.port = port
        self.model_name = model_name
        self.project_name = project_name
        self.hf_token = hf_token
        self.gpu_ids = gpu_ids
        self.verbose = verbose

        # Create compose manager
        self._manager = None
        if model_name:
            self._manager = TEIComposeManager(
                model_name=model_name,
                port=port,
                project_name=project_name,
                gpu_ids=gpu_ids,
                hf_token=hf_token,
            )

    def _ensure_manager(self) -> bool:
        """Ensure manager is initialized."""
        if not self._manager:
            logger.warn("× Model name is required (-m)")
            return False
        return True

    def up(self):
        """Start TEI containers."""
        if not self._ensure_manager():
            return
        self._manager.up()
        if self.verbose:
            self._manager.logs(follow=True)

    def down(self):
        """Stop and remove TEI containers."""
        if not self._ensure_manager():
            return
        self._manager.down()

    def stop(self):
        """Stop TEI containers (keep containers)."""
        if not self._ensure_manager():
            return
        self._manager.stop()

    def start(self):
        """Start stopped TEI containers."""
        if not self._ensure_manager():
            return
        self._manager.start()

    def ps(self):
        """Show status of TEI containers."""
        if not self._ensure_manager():
            return
        self._manager.ps()

    def logs(self, follow: bool = False, tail: int = 100):
        """Show logs of TEI containers."""
        if not self._ensure_manager():
            return
        self._manager.logs(follow=follow, tail=tail)

    def restart(self):
        """Restart TEI containers."""
        if not self._ensure_manager():
            return
        self._manager.restart()


class TEIServerByConfig(TEIServer):
    def __init__(self, configs: TEIServerConfigsType):
        super().__init__(**configs)


EPILOG = """
Examples:
  # Basic usage
  tei_server -m "Alibaba-NLP/gte-multilingual-base"           # Start on all GPUs
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -g "0,2"  # Start on specific GPUs
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -b        # Start and follow logs
  
  # Management operations
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -ps       # Show container status
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -l        # Show recent logs
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -l -f     # Follow logs
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -stop     # Stop containers
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -start    # Start stopped containers
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -r        # Restart containers
  tei_server -m "Alibaba-NLP/gte-multilingual-base" -down     # Stop and remove

Other models:
  tei_server -m "BAAI/bge-large-zh-v1.5" -p 28889
  tei_server -m "Qwen/Qwen3-Embedding-0.6B" -p 28887 -u hf_****

Note: For advanced Docker Compose operations, use 'tei_compose' command.
"""


class TEIServerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            description="TEI Server Manager (Docker Compose)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=EPILOG,
            **kwargs,
        )
        # Model config
        self.add_argument(
            "-m", "--model-name", type=str, default=None, help="HuggingFace model name"
        )
        self.add_argument(
            "-n",
            "--project-name",
            type=str,
            default=None,
            help="Docker compose project name (default: tei--MODEL_NAME)",
        )
        self.add_argument(
            "-u",
            "--hf-token",
            type=str,
            default=None,
            help="HuggingFace token for private models",
        )
        self.add_argument(
            "-p",
            "--port",
            type=int,
            default=SERVER_PORT,
            help=f"Starting port (default: {SERVER_PORT})",
        )
        self.add_argument(
            "-g",
            "--gpu-ids",
            type=str,
            default=None,
            help="Comma-separated GPU IDs (e.g., '0,1,2'). Default: all GPUs",
        )
        self.add_argument(
            "-b", "--verbose", action="store_true", help="Follow logs after starting"
        )
        # Actions (mutually exclusive operations)
        self.add_argument(
            "-down",
            "--down",
            action="store_true",
            help="Stop and remove containers",
        )
        self.add_argument(
            "-stop", "--stop", action="store_true", help="Stop containers (keep them)"
        )
        self.add_argument(
            "-start",
            "--start",
            action="store_true",
            help="Start stopped containers",
        )
        self.add_argument(
            "-ps",
            "--ps",
            action="store_true",
            help="Show container status",
        )
        self.add_argument(
            "-l", "--logs", action="store_true", help="Show container logs"
        )
        self.add_argument(
            "-f", "--follow", action="store_true", help="Follow logs (use with -l)"
        )
        self.add_argument(
            "-r", "--restart", action="store_true", help="Restart containers"
        )

        self.args, _ = self.parse_known_args()


def main():
    args = TEIServerArgParser().args

    tei_server = TEIServer(
        port=args.port,
        model_name=args.model_name,
        project_name=args.project_name,
        hf_token=args.hf_token,
        gpu_ids=args.gpu_ids,
        verbose=args.verbose,
    )

    if args.down:
        tei_server.down()
    elif args.stop:
        tei_server.stop()
    elif args.start:
        tei_server.start()
    elif args.ps:
        tei_server.ps()
    elif args.logs:
        tei_server.logs(follow=args.follow)
    elif args.restart:
        tei_server.restart()
    else:
        tei_server.up()


if __name__ == "__main__":
    main()
