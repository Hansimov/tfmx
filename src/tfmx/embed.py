import requests
import numpy as np

from tclogger import StrsType, logger
from typing import Literal, TypedDict, Optional, Union

EmbedApiType = Literal["openai", "tei"]


class EmbedConfigsType(TypedDict):
    endpoint: str
    api_key: Optional[str]
    model: str
    api_format: EmbedApiType = "tei"


class EmbedClient:
    def __init__(
        self,
        endpoint: str,
        model: str = None,
        api_key: str = None,
        api_format: EmbedApiType = "tei",
        output_format: Literal["list2d", "ndarray"] = "ndarray",
        verbose: bool = False,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.api_format = api_format
        self.output_format = output_format
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
        if self.output_format == "ndarray":
            return np.array(embeddings)
        else:
            return embeddings


class EmbedClientByConfig(EmbedClient):
    def __init__(self, configs: EmbedConfigsType):
        super().__init__(**configs)
