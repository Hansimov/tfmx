"""QWN client helpers for single-endpoint text chat services."""

import argparse
import httpx
import orjson

from dataclasses import dataclass, field
from tclogger import dict_to_lines, logger, logstr
from typing import Optional

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
    content: str

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
class InstanceInfo:
    name: str
    endpoint: str
    gpu_id: Optional[int]
    healthy: bool
    model_name: str = ""
    quant_method: str = ""
    quant_level: str = ""
    model_label: str = ""

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
    ) -> ChatResponse:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if model:
            payload["model"] = model

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

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model: str = "",
    ) -> str:
        messages = build_text_messages(prompt=prompt, system_prompt=system_prompt)
        response = self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
    ) -> ChatResponse:
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if model:
            payload["model"] = model

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
    ) -> str:
        messages = build_text_messages(prompt=prompt, system_prompt=system_prompt)
        response = await self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.text


CLI_EPILOG = """
Examples:
  qwn client health
  qwn client models
  qwn client chat "你好，介绍一下自己"
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
    chat_parser.add_argument("text", help="Prompt text")
    chat_parser.add_argument("--model", default="", help="Model label")
    chat_parser.add_argument("--max-tokens", type=int, default=512)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument("--top-p", type=float, default=0.9)
    chat_parser.add_argument("-s", "--system", default=None, help="System prompt")

    generate_parser = subparsers.add_parser(
        "generate", help="Generate text from a prompt"
    )
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--model", default="", help="Model label")
    generate_parser.add_argument("--max-tokens", type=int, default=512)
    generate_parser.add_argument("--temperature", type=float, default=0.7)
    generate_parser.add_argument("--top-p", type=float, default=0.9)
    generate_parser.add_argument("-s", "--system", default=None, help="System prompt")


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
                logger.mesg(
                    f"  - {instance.name}: {instance.endpoint} ({instance.model_label})"
                )
            logger.note("Stats:")
            for line in dict_to_lines(info.stats.__dict__).splitlines():
                logger.mesg(f"  {line}")
        elif args.client_action == "chat":
            result = client.generate(
                prompt=args.text,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                model=args.model,
            )
            print(result)
        elif args.client_action == "generate":
            result = client.generate(
                prompt=args.prompt,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                model=args.model,
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
