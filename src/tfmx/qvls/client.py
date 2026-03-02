"""QVL (Qwen3-VL) Client

Provides a client for connecting to vLLM services serving Qwen3-VL models,
using the OpenAI-compatible API for chat completions with vision support.
"""

# ANCHOR[id=client-clis]
CLI_EPILOG = """
Examples:
  # Connect to qvl_machine (default port 29800)
  qvl_client health                         # Check health
  qvl_client models                         # List models
  qvl_client chat "Hello, how are you?"     # Simple text chat
  qvl_client chat -i photo.jpg "Describe"   # Chat with image

  # Connect to specific endpoint
  qvl_client -e "http://localhost:29800" health
  qvl_client -e "http://localhost:29880" chat "Hello"

  # With custom port
  qvl_client -p 29800 health

  # Chat with parameters
  qvl_client chat --max-tokens 256 --temperature 0.7 "Tell me a story"
"""

import argparse
import base64
import httpx
import json
import orjson
import time

from dataclasses import dataclass, field
from pathlib import Path
from tclogger import logger, logstr, dict_to_lines
from typing import Optional, Union

PORT = 29800  # default port for qvl_machine
HOST = "localhost"


@dataclass
class HealthResponse:
    """Health check response."""

    status: str
    healthy: int
    total: int

    @classmethod
    def from_dict(cls, data: dict) -> "HealthResponse":
        if isinstance(data, str):
            # vLLM returns empty string for healthy
            return cls(status="healthy", healthy=1, total=1)
        return cls(
            status=data.get("status", "unknown"),
            healthy=data.get("healthy", 0),
            total=data.get("total", 0),
        )


@dataclass
class ModelInfo:
    """Information about available models."""

    models: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        models = []
        for model_data in data.get("data", []):
            models.append(model_data.get("id", "unknown"))
        return cls(models=models)


@dataclass
class ChatMessage:
    """A chat message."""

    role: str
    content: Union[str, list[dict]]

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
    """Token usage information."""

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
    """A single chat completion choice."""

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
    """Chat completion response."""

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
            choices=[ChatChoice.from_dict(c) for c in data.get("choices", [])],
            usage=ChatUsage.from_dict(data.get("usage", {})),
            created=data.get("created", 0),
        )

    @property
    def text(self) -> str:
        """Get the text content of the first choice."""
        if self.choices:
            content = self.choices[0].message.content
            return content if isinstance(content, str) else str(content)
        return ""


@dataclass
class InstanceInfo:
    """Information about a single vLLM instance."""

    name: str
    endpoint: str
    gpu_id: Optional[int]
    healthy: bool

    @classmethod
    def from_dict(cls, data: dict) -> "InstanceInfo":
        return cls(
            name=data.get("name", ""),
            endpoint=data.get("endpoint", ""),
            gpu_id=data.get("gpu_id"),
            healthy=data.get("healthy", False),
        )


@dataclass
class MachineStats:
    """Statistics for the machine."""

    total_requests: int
    total_tokens: int
    total_errors: int
    requests_per_instance: dict[str, int]

    @classmethod
    def from_dict(cls, data: dict) -> "MachineStats":
        return cls(
            total_requests=data.get("total_requests", 0),
            total_tokens=data.get("total_tokens", 0),
            total_errors=data.get("total_errors", 0),
            requests_per_instance=data.get("requests_per_instance", {}),
        )


def _encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 data URL."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Determine MIME type
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{data}"


def build_vision_messages(
    prompt: str,
    images: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[dict]:
    """Build OpenAI-format messages with vision support.

    Args:
        prompt: Text prompt
        images: List of image paths or URLs
        system_prompt: Optional system message

    Returns:
        List of message dicts in OpenAI format
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content_parts = []

    # Add images
    if images:
        for img in images:
            if img.startswith(("http://", "https://", "data:")):
                url = img
            else:
                url = _encode_image_to_base64(img)
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

    # Add text
    content_parts.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": content_parts})
    return messages


class QVLClient:
    """Synchronous client for QVL (Qwen3-VL) services.

    Can connect to either:
    - qvl_machine (load-balanced proxy, default port 29800)
    - vLLM containers (direct, ports 29880+)

    Example:
        client = QVLClient("http://localhost:29800")
        resp = client.chat([{"role": "user", "content": "Hello"}])
        print(resp.text)
    """

    def __init__(
        self,
        endpoint: str = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
    ):
        if endpoint:
            self.endpoint = endpoint.rstrip("/")
        else:
            self.endpoint = f"http://{host}:{port}"

        self.verbose = verbose
        self.client = httpx.Client(timeout=httpx.Timeout(120.0))

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def __enter__(self) -> "QVLClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _log_fail(self, action: str, error: Exception) -> None:
        if self.verbose:
            logger.warn(f"× QVL {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        if self.verbose:
            logger.okay(f"✓ QVL {action}: {message}")

    def _extract_error_detail(self, e: httpx.HTTPStatusError) -> str:
        try:
            return e.response.json().get("detail", str(e))
        except Exception:
            return str(e)

    def health(self) -> HealthResponse:
        """Check health status of the service."""
        try:
            resp = self.client.get(f"{self.endpoint}/health")
            resp.raise_for_status()
            # vLLM returns empty body for healthy
            try:
                data = resp.json()
            except Exception:
                data = {"status": "healthy", "healthy": 1, "total": 1}

            if isinstance(data, dict):
                result = HealthResponse.from_dict(data)
            else:
                result = HealthResponse(status="healthy", healthy=1, total=1)
            self._log_okay("health", f"status={result.status}")
            return result
        except httpx.HTTPStatusError as e:
            try:
                data = e.response.json()
                if "detail" in data and isinstance(data["detail"], dict):
                    return HealthResponse.from_dict(data["detail"])
            except Exception:
                pass
            self._log_fail("health", e)
            raise
        except Exception as e:
            self._log_fail("health", e)
            raise

    def is_healthy(self) -> bool:
        try:
            result = self.health()
            return result.status == "healthy" or result.healthy > 0
        except Exception:
            return False

    def log_machine_health(self) -> None:
        health = self.health()
        logger.mesg(f"* Healthy: {logstr.okay(health.healthy)}/{health.total}")

    def models(self) -> ModelInfo:
        """List available models."""
        try:
            resp = self.client.get(f"{self.endpoint}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            result = ModelInfo.from_dict(data)
            self._log_okay("models", f"count={len(result.models)}")
            return result
        except Exception as e:
            self._log_fail("models", e)
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
        """Send a chat completion request.

        Args:
            messages: List of message dicts in OpenAI format
            model: Model name (usually auto-detected)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stream: Whether to stream the response

        Returns:
            ChatResponse with generated text
        """
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
            resp = self.client.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            result = ChatResponse.from_dict(data)
            self._log_okay(
                "chat",
                f"tokens={result.usage.total_tokens}, finish={result.choices[0].finish_reason if result.choices else 'none'}",
            )
            return result
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e)
            self._log_fail("chat", error_detail)
            raise ValueError(f"Chat failed: {error_detail}") from e
        except Exception as e:
            self._log_fail("chat", e)
            raise

    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        model: str = "",
    ) -> str:
        """Convenience method for vision-language generation.

        Args:
            prompt: Text prompt
            images: Optional list of image paths or URLs
            system_prompt: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model: Model name

        Returns:
            Generated text string
        """
        messages = build_vision_messages(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
        )
        response = self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.text


class AsyncQVLClient:
    """Asynchronous client for QVL services.

    Example:
        async with AsyncQVLClient("http://localhost:29800") as client:
            resp = await client.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        endpoint: str = None,
        host: str = HOST,
        port: int = PORT,
        verbose: bool = False,
    ):
        if endpoint:
            self.endpoint = endpoint.rstrip("/")
        else:
            self.endpoint = f"http://{host}:{port}"

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
        """Reset the client state for use in a new event loop."""
        self._client = None

    async def __aenter__(self) -> "AsyncQVLClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def _log_fail(self, action: str, error: Exception) -> None:
        if self.verbose:
            logger.warn(f"× AsyncQVL {action} error: {error}")

    def _log_okay(self, action: str, message: str) -> None:
        if self.verbose:
            logger.okay(f"✓ AsyncQVL {action}: {message}")

    def _extract_error_detail(self, e: httpx.HTTPStatusError) -> str:
        try:
            return e.response.json().get("detail", str(e))
        except Exception:
            return str(e)

    async def health(self) -> HealthResponse:
        """Check health status."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.endpoint}/health")
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                data = {"status": "healthy", "healthy": 1, "total": 1}

            if isinstance(data, dict):
                result = HealthResponse.from_dict(data)
            else:
                result = HealthResponse(status="healthy", healthy=1, total=1)
            self._log_okay("health", f"status={result.status}")
            return result
        except httpx.HTTPStatusError as e:
            try:
                data = e.response.json()
                if "detail" in data and isinstance(data["detail"], dict):
                    return HealthResponse.from_dict(data["detail"])
            except Exception:
                pass
            self._log_fail("health", e)
            raise
        except Exception as e:
            self._log_fail("health", e)
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
        """Send a chat completion request."""
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
            content = orjson.dumps(payload)
            resp = await client.post(
                f"{self.endpoint}/v1/chat/completions",
                content=content,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = orjson.loads(resp.content)
            result = ChatResponse.from_dict(data)
            self._log_okay(
                "chat",
                f"tokens={result.usage.total_tokens}",
            )
            return result
        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e)
            self._log_fail("chat", error_detail)
            raise ValueError(f"Chat failed: {error_detail}") from e
        except Exception as e:
            self._log_fail("chat", e)
            raise

    async def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        model: str = "",
    ) -> str:
        """Convenience method for vision-language generation."""
        messages = build_vision_messages(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
        )
        response = await self.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.text


class QVLClientArgParser:
    """Argument parser for QVL Client CLI."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="QVL Client - Connect to Qwen3-VL services",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        self.parser.add_argument(
            "-e",
            "--endpoint",
            type=str,
            default=None,
            help="Full endpoint URL",
        )
        self.parser.add_argument(
            "-p",
            "--port",
            type=int,
            default=PORT,
            help=f"Server port (default: {PORT})",
        )
        self.parser.add_argument(
            "-H",
            "--host",
            type=str,
            default=HOST,
            help=f"Server host (default: {HOST})",
        )
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

        subparsers = self.parser.add_subparsers(dest="action", help="Action to perform")

        subparsers.add_parser("health", help="Check service health")
        subparsers.add_parser("models", help="List available models")

        chat_parser = subparsers.add_parser("chat", help="Send chat completion")
        chat_parser.add_argument("text", help="Text prompt")
        chat_parser.add_argument(
            "-i",
            "--images",
            nargs="*",
            default=None,
            help="Image paths or URLs",
        )
        chat_parser.add_argument(
            "--max-tokens",
            type=int,
            default=512,
            help="Max tokens to generate (default: 512)",
        )
        chat_parser.add_argument(
            "--temperature",
            type=float,
            default=0.7,
            help="Sampling temperature (default: 0.7)",
        )
        chat_parser.add_argument(
            "-s",
            "--system",
            type=str,
            default=None,
            help="System prompt",
        )


class QVLClientCLI:
    """CLI interface for QVL Client operations."""

    def __init__(self, client: QVLClient):
        self.client = client

    def run_health(self) -> None:
        self.client.log_machine_health()

    def run_models(self) -> None:
        info = self.client.models()
        logger.note("Available models:")
        for model in info.models:
            logger.mesg(f"  - {model}")

    def run_chat(
        self,
        text: str,
        images: list[str] | None = None,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        result = self.client.generate(
            prompt=text,
            images=images,
            system_prompt=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(result)


def main():
    arg_parser = QVLClientArgParser()
    args = arg_parser.args

    if args.action is None:
        arg_parser.parser.print_help()
        return

    client = QVLClient(
        endpoint=args.endpoint,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

    try:
        cli = QVLClientCLI(client)

        if args.action == "health":
            cli.run_health()
        elif args.action == "models":
            cli.run_models()
        elif args.action == "chat":
            cli.run_chat(
                text=args.text,
                images=getattr(args, "images", None),
                system=getattr(args, "system", None),
                max_tokens=getattr(args, "max_tokens", 512),
                temperature=getattr(args, "temperature", 0.7),
            )
    except httpx.ConnectError as e:
        logger.warn(f"× Connection failed: {e}")
        logger.hint(f"  Is the QVL service running at {client.endpoint}?")
    except Exception as e:
        logger.warn(f"× Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
