import numpy as np
import requests

from tclogger import StrsType, logger
from typing import Literal, TypedDict, Optional, Union

EmbedApiFormat = Literal["openai", "tei"]
EmbedResFormat = Literal["ndarray", "list2d"]


class EmbedClientConfigsType(TypedDict):
    endpoint: str
    api_key: Optional[str]
    model: str
    api_format: EmbedApiFormat = "tei"
    res_format: EmbedResFormat = "list2d"
    batch_size: int = 100


class EmbedClient:
    def __init__(
        self,
        endpoint: str,
        model: str = None,
        api_key: str = None,
        api_format: EmbedApiFormat = "tei",
        res_format: EmbedResFormat = "list2d",
        batch_size: int = 100,
        verbose: bool = False,
    ):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.api_format = api_format
        self.res_format = res_format
        self.batch_size = batch_size
        self.verbose = verbose

    def log_resp_status(self, resp: requests.Response):
        err_mesg = f"× Embed error: {resp.status_code} {resp.text}"
        if self.verbose:
            logger.warn(err_mesg)
        raise ValueError(err_mesg)

    def log_embed_res(self, embeddings: list[list[float]]):
        if self.verbose:
            num = len(embeddings)
            dim = len(embeddings[0]) if num > 0 else 0
            val_type = type(embeddings[0][0]).__name__ if dim > 0 else "N/A"
            logger.okay(f"✓ Embed success: num={num}, dim={dim}, type={val_type}")

    def embed_batch(self, inputs: StrsType) -> list[list[float]]:
        headers = {
            "content-type": "application/json",
        }
        # https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
        payload = {
            "inputs": inputs,
            "normalize": True,
            "truncate": True,
        }
        resp = requests.post(self.endpoint, headers=headers, json=payload)
        if resp.status_code != 200:
            self.log_resp_status(resp)
            return []
        embeddings = resp.json()
        self.log_embed_res(embeddings)
        return embeddings

    def embed(self, inputs: StrsType) -> Union[list[list[float]], np.ndarray]:
        if isinstance(inputs, str):
            inputs = [inputs]
        if len(inputs) <= self.batch_size:
            return self.embed_batch(inputs)
        embs = []
        for i in range(0, len(inputs), self.batch_size):
            inputs_batch = inputs[i : i + self.batch_size]
            embs_batch = self.embed_batch(inputs_batch)
            embs.extend(embs_batch)
        if self.res_format == "ndarray":
            return np.array(embs)
        else:
            return embs


class EmbedClientByConfig(EmbedClient):
    def __init__(self, configs: EmbedClientConfigsType):
        super().__init__(**configs)
