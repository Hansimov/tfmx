import argparse
import shlex

from pathlib import Path
from tclogger import logger, shell_cmd
from typing import TypedDict


class TEIEmbedServerConfigsType(TypedDict):
    port: int
    model_name: str
    instance_id: str
    verbose: bool


class TEIEmbedServer:
    def __init__(
        self,
        port: int = None,
        model_name: str = None,
        instance_id: str = None,
        verbose: bool = False,
    ):
        self.port = port
        self.model_name = model_name
        self.instance_id = instance_id
        self.verbose = verbose

    def run(self):
        script_path = Path(__file__).resolve().parent / "run_tei.sh"
        if not script_path.exists():
            logger.warn(f"× Missing `run_tei.sh`: {script_path}")
            return

        run_parts = ["bash", str(script_path)]
        if self.port:
            run_parts.extend(["-p", str(self.port)])
        if self.model_name:
            run_parts.extend(["-m", self.model_name])
        if self.instance_id:
            run_parts.extend(["-id", self.instance_id])
        cmd_run = shlex.join(run_parts)
        shell_cmd(cmd_run)

        if self.verbose:
            cmd_logs = f'docker logs -f "{self.instance_id}"'
            shell_cmd(cmd_logs)

    def kill(self):
        if not self.instance_id:
            logger.warn("× Missing arg: -id (--instance-id)")
            return

        cmd_kill = f'docker stop "{self.instance_id}"'
        shell_cmd(cmd_kill)


class TEIEmbedServerByConfig(TEIEmbedServer):
    def __init__(self, configs: TEIEmbedServerConfigsType):
        super().__init__(**configs)


class TEIEmbedServerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-p", "--port", type=int, default=28888)
        self.add_argument(
            "-m",
            "--model-name",
            type=str,
            default="Alibaba-NLP/gte-multilingual-base",
        )
        self.add_argument(
            "-id",
            "--instance-id",
            type=str,
            default="Alibaba-NLP--gte-multilingual-base",
        )
        self.add_argument("-k", "--kill", action="store_true")
        self.add_argument("-b", "--verbose", action="store_true")
        self.args, _ = self.parse_known_args()


class EmbedServerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-t", "--type", type=str, choices=["tei"], default="tei")
        self.args, _ = self.parse_known_args()


def main():
    main_args = EmbedServerArgParser().args
    if main_args.type == "tei":
        args = TEIEmbedServerArgParser().args
        embed_server = TEIEmbedServer(
            port=args.port,
            model_name=args.model_name,
            instance_id=args.instance_id,
            verbose=args.verbose,
        )
        if args.kill:
            embed_server.kill()
        else:
            embed_server.run()


if __name__ == "__main__":
    main()

    # Case 1: gte-multilingual-base
    # python -m tfmx.embed_server -t "tei" -p 28888 -m "Alibaba-NLP/gte-multilingual-base" -id "Alibaba-NLP--gte-multilingual-base" -b
    # python -m tfmx.embed_server -t "tei" -id "Alibaba-NLP--gte-multilingual-base" -k

    # Case 2: bge-large-zh-v1.5
    # python -m tfmx.embed_server -t "tei" -p 28889 -m "BAAI/bge-large-zh-v1.5" -id "BAAI--bge-large-zh-v1.5" -b
    # python -m tfmx.embed_server -t "tei" -id "BAAI--bge-large-zh-v1.5" -k
