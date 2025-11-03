import argparse
import numpy as np
import requests
import shlex

from pathlib import Path
from tclogger import StrsType, logger, shell_cmd
from typing import Literal, TypedDict, Optional, Union

EmbedApiFormat = Literal["openai", "tei"]
EmbedResFormat = Literal["ndarray", "list2d"]


class EmbedConfigsType(TypedDict):
    endpoint: str
    api_key: Optional[str]
    model: str
    api_format: EmbedApiFormat = "tei"
    res_format: EmbedResFormat = "list2d"


class EmbedClient:
    def __init__(
        self,
        endpoint: str,
        model: str = None,
        api_key: str = None,
        api_format: EmbedApiFormat = "tei",
        res_format: EmbedResFormat = "list2d",
        verbose: bool = False,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.api_format = api_format
        self.res_format = res_format
        self.verbose = verbose

    def log_resp_status(self, resp: requests.Response):
        if self.verbose:
            logger.warn(f"× Embed error: {resp.status_code} {resp.text}")

    def log_embed_res(self, embeddings: list[list[float]]):
        if self.verbose:
            num = len(embeddings)
            dim = len(embeddings[0]) if num > 0 else 0
            val_type = type(embeddings[0][0]).__name__ if dim > 0 else "N/A"
            logger.okay(f"✓ Embed success: num={num}, dim={dim}, type={val_type}")

    def embed(self, inputs: StrsType) -> Union[list[list[float]], np.ndarray]:
        headers = {
            "content-type": "application/json",
        }
        payload = {
            "inputs": inputs,
        }
        resp = requests.post(self.endpoint, headers=headers, json=payload)
        if resp.status_code != 200:
            self.log_resp_status(resp)
            return []
        embeddings = resp.json()
        self.log_embed_res(embeddings)
        if self.res_format == "ndarray":
            return np.array(embeddings)
        else:
            return embeddings


class EmbedClientByConfig(EmbedClient):
    def __init__(self, configs: EmbedConfigsType):
        super().__init__(**configs)


class EmbedServerByTEI:
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


class EmbedServerByTEIArgParser(argparse.ArgumentParser):
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
        args = EmbedServerByTEIArgParser().args
        embed_server = EmbedServerByTEI(
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

    # python -m tfmx.embed -t "tei" -p 28888 -m "Alibaba-NLP/gte-multilingual-base" -id "Alibaba-NLP--gte-multilingual-base" -b
    # python -m tfmx.embed -t "tei" -id "Alibaba-NLP--gte-multilingual-base" -k
