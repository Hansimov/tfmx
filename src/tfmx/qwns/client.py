"""QWN client helpers for single-endpoint text chat services."""

import argparse
import base64
import time

import httpx
import orjson

from dataclasses import dataclass, field
from pathlib import Path
from tclogger import dict_to_lines, logger, logstr
from typing import Callable, Optional

from .compose import MACHINE_PORT as PORT


HOST = "localhost"


@dataclass
class HealthResponse:
    status: str
    healthy: int
    total: int

    @classmethod
    def from_dict(cls, data: dict | str) -> "HealthResponse":
        if isinstance(data, str):
            return cls(status="healthy", healthy=1, total=1)
        return cls(
            status=data.get("status", "unknown"),
            healthy=data.get("healthy", 0),
            total=data.get("total", 0),
        )


@dataclass
class ModelInfo:
    models: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        return cls(models=[item.get("id", "") for item in data.get("data", [])])


@dataclass
class ChatMessage:
    role: str
    content: object

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
        )


@dataclass
class ChatUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "ChatUsage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )


@dataclass
class ChatChoice:
    index: int
    message: ChatMessage
    finish_reason: str

    @classmethod
    def from_dict(cls, data: dict) -> "ChatChoice":
        return cls(
            index=data.get("index", 0),
            message=ChatMessage.from_dict(data.get("message", {})),
            finish_reason=data.get("finish_reason", "stop"),
        )


@dataclass
class ChatResponse:
    id: str
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
    created: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "ChatResponse":
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=[
                ChatChoice.from_dict(choice) for choice in data.get("choices", [])
            ],
            usage=ChatUsage.from_dict(data.get("usage", {})),
            created=data.get("created", 0),
        )

    @property
    def text(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content


@dataclass
class StreamChatResult:
    text: str = ""
    usage: ChatUsage = field(default_factory=ChatUsage)
    elapsed_sec: float = 0.0
    first_token_latency_sec: float = 0.0

    @property
    def token_per_second(self) -> float:
        if self.elapsed_sec <= 0:
            return 0.0
        token_count = self.usage.completion_tokens or self.usage.total_tokens
        return token_count / self.elapsed_sec if token_count > 0 else 0.0


@dataclass
class InstanceInfo:
    name: str
    endpoint: str
    gpu_id: Optional[int]
    healthy: bool
    model_name: str = ""
    quant_method: str = ""
    quant_level: str = ""
    model_label: str = ""
    gpu_utilization_pct: float | None = None
    gpu_memory_used_mib: float | None = None
    gpu_memory_total_mib: float | None = None
    routing_pressure: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "InstanceInfo":
        return cls(
            name=data.get("name", ""),
            endpoint=data.get("endpoint", ""),
            gpu_id=data.get("gpu_id"),
            healthy=data.get("healthy", False),
            model_name=data.get("model_name", ""),
            quant_method=data.get("quant_method", ""),
            quant_level=data.get("quant_level", ""),
            model_label=data.get("model_label", ""),
            gpu_utilization_pct=data.get("gpu_utilization_pct"),
            gpu_memory_used_mib=data.get("gpu_memory_used_mib"),
            gpu_memory_total_mib=data.get("gpu_memory_total_mib"),
            routing_pressure=data.get("routing_pressure"),
        )


@dataclass
class MachineStats:
    total_requests: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    active_requests: int = 0
    requests_per_instance: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "MachineStats":
        return cls(
            total_requests=data.get("total_requests", 0),
            total_tokens=data.get("total_tokens", 0),
            total_errors=data.get("total_errors", 0),
            active_requests=data.get("active_requests", 0),
            requests_per_instance=data.get("requests_per_instance", {}),
        )


@dataclass
class InfoResponse:
    port: int
    instances: list[InstanceInfo] = field(default_factory=list)
    stats: MachineStats = field(default_factory=MachineStats)
    available_models: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "InfoResponse":
        return cls(
            port=data.get("port", 0),
            instances=[
                InstanceInfo.from_dict(item) for item in data.get("instances", [])
            ],
            stats=MachineStats.from_dict(data.get("stats", {})),
            available_models=data.get("available_models", []),
        )


def build_text_messages(
    prompt: str,
    system_prompt: str | None = None,
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def join_prompt_texts(texts: list[str] | tuple[str, ...] | str) -> str:
    if isinstance(texts, str):
        return texts.strip()
    return "\n\n".join(part.strip() for part in texts if part and part.strip())


def _encode_image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(path.suffix.lower(), "image/png")
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def normalize_image_url(image: str) -> str:
    if image.startswith(("http://", "https://", "data:")):
        return image
    return _encode_image_to_data_url(image)


def build_multimodal_messages(
    texts: list[str],
    images: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    normalized_texts = [text.strip() for text in texts if text and text.strip()]
    normalized_images = [normalize_image_url(image) for image in images or []]

    if not normalized_images:
        return build_text_messages(join_prompt_texts(normalized_texts), system_prompt)

    content_parts: list[dict] = []
    total_parts = max(len(normalized_texts), len(normalized_images))
    for index in range(total_parts):
        if index < len(normalized_texts):
            content_parts.append({"type": "text", "text": normalized_texts[index]})
        if index < len(normalized_images):
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": normalized_images[index]},
                }
            )

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})
    return messages


def _build_chat_payload(
    messages: list[dict],
    model: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stream: bool = False,
    enable_thinking: bool = False,
) -> dict:
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if model:
        payload["model"] = model
    return payload


def format_elapsed_time(elapsed_sec: float) -> str:
    total_seconds = max(0.0, elapsed_sec)
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    if minutes > 0:
        return f"{minutes}min {seconds:.1f}s"
    return f"{seconds:.1f}s"


def format_stream_stats_line(result: StreamChatResult) -> str:
    elapsed_text = logstr.okay(format_elapsed_time(result.elapsed_sec))
    ttft_value = (
        format_elapsed_time(result.first_token_latency_sec)
        if result.first_token_latency_sec > 0
        else "n/a"
    )
    ttft_text = (
        logstr.mesg(ttft_value)
        if result.first_token_latency_sec > 0
        else logstr.warn(ttft_value)
    )
    rate_value = f"{result.token_per_second:.1f} token/s"
    rate_text = (
        logstr.okay(rate_value)
        if result.token_per_second > 0
        else logstr.mesg(rate_value)
    )
    token_count = result.usage.completion_tokens or result.usage.total_tokens
    token_text = logstr.mesg(f"{token_count} tok")
    return (
        f"{logstr.mesg('stats')} elapsed={elapsed_text} | "
        f"ttft={ttft_text} | rate={rate_text} | out={token_text}"
    )


def _iter_text_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return parts
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return [text]
    return []


def _extract_stream_text(chunk: dict) -> str:
    fragments: list[str] = []
    for choice in chunk.get("choices", []):
        delta = choice.get("delta") or choice.get("message") or {}
        fragments.extend(_iter_text_parts(delta.get("content")))
    return "".join(fragments)


class QWNClient:
    def __init__(
        self,
        endpoint: str | None = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
    ):
        self.endpoint = endpoint.rstrip("/") if endpoint else f"http://{host}:{port}"
        self.verbose = verbose
        self.client = httpx.Client(timeout=httpx.Timeout(120.0))

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self) -> "QWNClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _log_fail(self, action: str, error: Exception | str) -> None:
        if self.verbose:
            logger.warn(f"× QWN {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        if self.verbose:
            logger.okay(f"✓ QWN {action}: {message}")

    def _extract_error_detail(self, exc: httpx.HTTPStatusError) -> str:
        try:
            payload = exc.response.json()
            if "error" in payload:
                return payload["error"].get("message", str(exc))
            if "detail" in payload:
                return str(payload["detail"])
            return str(payload)
        except Exception:
            return str(exc)

    def health(self) -> HealthResponse:
        try:
            response = self.client.get(f"{self.endpoint}/health")
            response.raise_for_status()
            try:
                payload = response.json()
            except Exception:
                payload = {"status": "healthy", "healthy": 1, "total": 1}
            result = HealthResponse.from_dict(payload)
            self._log_okay("health", f"status={result.status}")
            return result
        except httpx.HTTPStatusError as exc:
            try:
                payload = exc.response.json()
                if "detail" in payload and isinstance(payload["detail"], dict):
                    return HealthResponse.from_dict(payload["detail"])
            except Exception:
                pass
            self._log_fail("health", exc)
            raise
        except Exception as exc:
            self._log_fail("health", exc)
            raise

    def log_machine_health(self) -> None:
        health = self.health()
        logger.mesg(f"* Healthy: {logstr.okay(health.healthy)}/{health.total}")

    def is_healthy(self) -> bool:
        try:
            health = self.health()
            return health.status == "healthy" or health.healthy > 0
        except Exception:
            return False

    def models(self) -> ModelInfo:
        try:
            response = self.client.get(f"{self.endpoint}/v1/models")
            response.raise_for_status()
            result = ModelInfo.from_dict(response.json())
            self._log_okay("models", f"count={len(result.models)}")
            return result
        except Exception as exc:
            self._log_fail("models", exc)
            raise

    def info(self) -> InfoResponse:
        try:
            response = self.client.get(f"{self.endpoint}/info")
            response.raise_for_status()
            result = InfoResponse.from_dict(response.json())
            self._log_okay("info", f"instances={len(result.instances)}")
            return result
        except Exception as exc:
            self._log_fail("info", exc)
            raise

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> ChatResponse:
        payload = _build_chat_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            enable_thinking=enable_thinking,
        )

        try:
            response = self.client.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            result = ChatResponse.from_dict(response.json())
            self._log_okay("chat", f"tokens={result.usage.total_tokens}")
            return result
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc)
            self._log_fail("chat", detail)
            raise ValueError(f"Chat failed: {detail}") from exc
        except Exception as exc:
            self._log_fail("chat", exc)
            raise

    def stream_chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        on_text: Callable[[str], None] | None = None,
        enable_thinking: bool = False,
    ) -> StreamChatResult:
        payload = _build_chat_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            enable_thinking=enable_thinking,
        )
        payload["stream_options"] = {"include_usage": True}

        text_fragments: list[str] = []
        usage = ChatUsage()
        started_at = time.perf_counter()
        first_token_latency_sec = 0.0
        try:
            with self.client.stream(
                "POST",
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    chunk = orjson.loads(data)
                    if "error" in chunk:
                        detail = chunk["error"].get("message", str(chunk["error"]))
                        raise ValueError(f"Chat failed: {detail}")

                    chunk_text = _extract_stream_text(chunk)
                    if chunk_text:
                        if first_token_latency_sec <= 0:
                            first_token_latency_sec = time.perf_counter() - started_at
                        text_fragments.append(chunk_text)
                        if on_text:
                            on_text(chunk_text)

                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage = ChatUsage.from_dict(chunk_usage)

            elapsed_sec = time.perf_counter() - started_at
            result = StreamChatResult(
                text="".join(text_fragments),
                usage=usage,
                elapsed_sec=elapsed_sec,
                first_token_latency_sec=first_token_latency_sec,
            )
            self._log_okay("stream_chat", f"tokens={result.usage.total_tokens}")
            return result
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc)
            self._log_fail("stream_chat", detail)
            raise ValueError(f"Chat failed: {detail}") from exc
        except Exception as exc:
            self._log_fail("stream_chat", exc)
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "",
        enable_thinking: bool = False,
    ) -> str:
        messages = build_text_messages(prompt=prompt, system_prompt=system_prompt)
        response = self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        return response.text


class AsyncQWNClient:
    def __init__(
        self,
        endpoint: str | None = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
    ):
        self.endpoint = endpoint.rstrip("/") if endpoint else f"http://{host}:{port}"
        self.verbose = verbose
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def reset(self) -> None:
        self._client = None

    async def __aenter__(self) -> "AsyncQWNClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def _log_fail(self, action: str, error: Exception | str) -> None:
        if self.verbose:
            logger.warn(f"× AsyncQWN {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        if self.verbose:
            logger.okay(f"✓ AsyncQWN {action}: {message}")

    def _extract_error_detail(self, exc: httpx.HTTPStatusError) -> str:
        try:
            payload = exc.response.json()
            if "error" in payload:
                return payload["error"].get("message", str(exc))
            return str(payload)
        except Exception:
            return str(exc)

    async def health(self) -> HealthResponse:
        client = await self._get_client()
        try:
            response = await client.get(f"{self.endpoint}/health")
            response.raise_for_status()
            try:
                payload = response.json()
            except Exception:
                payload = {"status": "healthy", "healthy": 1, "total": 1}
            result = HealthResponse.from_dict(payload)
            self._log_okay("health", f"status={result.status}")
            return result
        except httpx.HTTPStatusError as exc:
            try:
                payload = exc.response.json()
                if "detail" in payload and isinstance(payload["detail"], dict):
                    return HealthResponse.from_dict(payload["detail"])
            except Exception:
                pass
            self._log_fail("health", exc)
            raise
        except Exception as exc:
            self._log_fail("health", exc)
            raise

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        enable_thinking: bool = False,
    ) -> ChatResponse:
        payload = _build_chat_payload(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            enable_thinking=enable_thinking,
        )

        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.endpoint}/v1/chat/completions",
                content=orjson.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = ChatResponse.from_dict(orjson.loads(response.content))
            self._log_okay("chat", f"tokens={result.usage.total_tokens}")
            return result
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc)
            self._log_fail("chat", detail)
            raise ValueError(f"Chat failed: {detail}") from exc
        except Exception as exc:
            self._log_fail("chat", exc)
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "",
        enable_thinking: bool = False,
    ) -> str:
        messages = build_text_messages(prompt=prompt, system_prompt=system_prompt)
        response = await self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        return response.text


CLI_EPILOG = """
Examples:
  qwn client health
  qwn client models
    qwn client chat "你好，介绍一下自己"
    qwn client chat -i photo_a.png -i photo_b.png "先看第一张图" "再比较第二张图"
    qwn client chat "你好" --no-stream
  qwn client generate --prompt "请总结一下 Docker 部署步骤"
  qwn client -e http://localhost:27880 models
"""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Connect to QWN machine or direct vLLM endpoints"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = CLI_EPILOG

    parser.add_argument(
        "-e", "--endpoint", type=str, default=None, help="Full endpoint URL"
    )
    parser.add_argument(
        "-p", "--port", type=int, default=PORT, help=f"Server port (default: {PORT})"
    )
    parser.add_argument(
        "-H", "--host", type=str, default=HOST, help=f"Server host (default: {HOST})"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="client_action", required=True)
    subparsers.add_parser("health", help="Check service health")
    subparsers.add_parser("models", help="List available models")
    subparsers.add_parser("info", help="Show machine info")

    chat_parser = subparsers.add_parser("chat", help="Send a chat completion")
    chat_parser.add_argument("text", nargs="+", help="Prompt text segment(s)")
    chat_parser.add_argument("--model", default="", help="Model label")
    chat_parser.add_argument("--max-tokens", type=int, default=512)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument("--top-p", type=float, default=0.9)
    chat_parser.add_argument("-s", "--system", default=None, help="System prompt")
    chat_parser.add_argument(
        "-i",
        "--image",
        action="append",
        default=[],
        help="Image path, URL, or data URI. Repeat for multi-image prompts.",
    )
    chat_parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable model thinking-mode output",
    )
    chat_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming terminal output",
    )

    generate_parser = subparsers.add_parser(
        "generate", help="Generate text from a prompt"
    )
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--model", default="", help="Model label")
    generate_parser.add_argument("--max-tokens", type=int, default=512)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-p", type=float, default=0.9)
    generate_parser.add_argument("-s", "--system", default=None, help="System prompt")
    generate_parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable model thinking-mode output",
    )


def run_from_args(args: argparse.Namespace) -> None:
    client = QWNClient(
        endpoint=args.endpoint,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

    try:
        if args.client_action == "health":
            client.log_machine_health()
        elif args.client_action == "models":
            info = client.models()
            logger.note("Available models:")
            for model in info.models:
                logger.mesg(f"  - {model}")
        elif args.client_action == "info":
            info = client.info()
            logger.note(f"Port: {info.port}")
            logger.note("Instances:")
            for instance in info.instances:
                details = [instance.model_label] if instance.model_label else []
                if instance.gpu_id is not None:
                    details.append(f"GPU{instance.gpu_id}")
                if instance.gpu_utilization_pct is not None:
                    details.append(f"util={instance.gpu_utilization_pct:.0f}%")
                if (
                    instance.gpu_memory_used_mib is not None
                    and instance.gpu_memory_total_mib is not None
                    and instance.gpu_memory_total_mib > 0
                ):
                    details.append(
                        f"mem={instance.gpu_memory_used_mib:.0f}/{instance.gpu_memory_total_mib:.0f}MiB"
                    )
                if instance.routing_pressure is not None:
                    details.append(f"pressure={instance.routing_pressure:.2f}")
                detail_text = ", ".join(details)
                logger.mesg(f"  - {instance.name}: {instance.endpoint} ({detail_text})")
            logger.note("Stats:")
            for line in dict_to_lines(info.stats.__dict__).splitlines():
                logger.mesg(f"  {line}")
        elif args.client_action == "chat":
            messages = build_multimodal_messages(
                texts=args.text,
                images=args.image,
                system_prompt=args.system,
            )
            if args.no_stream:
                result = client.chat(
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    model=args.model,
                    enable_thinking=args.thinking,
                )
                print(result.text)
            else:
                stream_result = client.stream_chat(
                    messages=messages,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    on_text=lambda chunk: print(chunk, end="", flush=True),
                    enable_thinking=args.thinking,
                )
                if stream_result.text and not stream_result.text.endswith("\n"):
                    print()
                print(format_stream_stats_line(stream_result))
        elif args.client_action == "generate":
            result = client.generate(
                prompt=args.prompt,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                model=args.model,
                enable_thinking=args.thinking,
            )
            print(result)
        else:
            raise ValueError(f"Unknown client action: {args.client_action}")
    except httpx.ConnectError as exc:
        logger.warn(f"× Connection failed: {exc}")
        logger.hint(f"  Is the QWN service running at {client.endpoint}?")
    finally:
        client.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
