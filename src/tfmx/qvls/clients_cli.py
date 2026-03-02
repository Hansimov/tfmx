"""QVL Clients CLI - Command-line interface for QVL multi-machine operations.

Provides CLI commands for interacting with QVL services:
- health: Check health of all machines
- info: Get model info from all machines
- chat: Run interactive chat session
- generate: Generate text from prompt + optional images

Usage:
    qvl_clients health --endpoints http://host1:29800 http://host2:29800
    qvl_clients generate --endpoints http://host1:29800 --prompt "Describe image" --image img.jpg
"""

import argparse
import base64
import json
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, Sequence

from .client import ChatResponse, HealthResponse, build_vision_messages


class QVLClientsProtocol(Protocol):
    """Protocol for QVL clients - defines expected interface."""

    def health(self): ...
    def chat(self, messages: list[dict], **kwargs) -> ChatResponse: ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    def chat_batch(self, requests: list[dict], **kwargs) -> list[ChatResponse]: ...
    def generate_batch(self, prompts: list[str], **kwargs) -> list[str]: ...
    def info(self) -> list: ...
    def close(self) -> None: ...


class QVLClientsArgParserBase(ABC):
    """Base argument parser for QVL clients CLI."""

    def __init__(self, description: str = "QVL Clients CLI"):
        self.parser = argparse.ArgumentParser(description=description)
        self._add_common_args()
        self._add_subcommands()

    def _add_common_args(self) -> None:
        self.parser.add_argument(
            "--endpoints",
            nargs="+",
            required=True,
            help="Machine endpoints (e.g., http://host1:29800 http://host2:29800)",
        )
        self.parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose logging",
        )

    def _add_subcommands(self) -> None:
        subparsers = self.parser.add_subparsers(dest="command", help="Command")
        subparsers.required = True

        # health
        subparsers.add_parser("health", help="Check health of all machines")

        # info
        subparsers.add_parser("info", help="Get model info from all machines")

        # chat
        chat_parser = subparsers.add_parser("chat", help="Interactive chat")
        chat_parser.add_argument("--model", default="", help="Model name")
        chat_parser.add_argument(
            "--max-tokens", type=int, default=512, help="Max tokens"
        )
        chat_parser.add_argument(
            "--temperature", type=float, default=0.7, help="Temperature"
        )
        chat_parser.add_argument("--system-prompt", default=None, help="System prompt")

        # generate
        gen_parser = subparsers.add_parser("generate", help="Generate text")
        gen_parser.add_argument("--prompt", required=True, help="Text prompt")
        gen_parser.add_argument(
            "--images",
            nargs="*",
            default=None,
            help="Image file paths or URLs",
        )
        gen_parser.add_argument("--system-prompt", default=None, help="System prompt")
        gen_parser.add_argument("--model", default="", help="Model name")
        gen_parser.add_argument(
            "--max-tokens", type=int, default=512, help="Max tokens"
        )
        gen_parser.add_argument(
            "--temperature", type=float, default=0.7, help="Temperature"
        )

        # benchmark (quick inline benchmark)
        bench_parser = subparsers.add_parser("bench", help="Quick benchmark")
        bench_parser.add_argument(
            "--n", type=int, default=10, help="Number of requests"
        )
        bench_parser.add_argument(
            "--max-tokens", type=int, default=64, help="Max tokens per request"
        )

    def parse(self, args: Sequence[str] | None = None) -> argparse.Namespace:
        return self.parser.parse_args(args)

    @abstractmethod
    def create_client(self, args: argparse.Namespace) -> QVLClientsProtocol:
        """Create concrete client from args."""
        ...


class QVLClientsCLIBase(ABC):
    """Base CLI runner for QVL clients."""

    def __init__(self, arg_parser: QVLClientsArgParserBase):
        self.arg_parser = arg_parser

    def run(self, args: Sequence[str] | None = None) -> None:
        parsed = self.arg_parser.parse(args)
        client = self.arg_parser.create_client(parsed)

        try:
            command = parsed.command
            if command == "health":
                self._cmd_health(client)
            elif command == "info":
                self._cmd_info(client)
            elif command == "chat":
                self._cmd_chat(client, parsed)
            elif command == "generate":
                self._cmd_generate(client, parsed)
            elif command == "bench":
                self._cmd_bench(client, parsed)
            else:
                print(f"Unknown command: {command}")
                sys.exit(1)
        finally:
            client.close()

    def _cmd_health(self, client: QVLClientsProtocol) -> None:
        health = client.health()
        print(f"Status: {health.status}")
        print(f"  Healthy machines: {health.healthy_machines}/{health.total_machines}")
        print(
            f"  Healthy instances: {health.healthy_instances}/{health.total_instances}"
        )

    def _cmd_info(self, client: QVLClientsProtocol) -> None:
        info = client.info()
        for i, models in enumerate(info):
            print(f"Machine {i}: {models}")

    def _cmd_chat(self, client: QVLClientsProtocol, args: argparse.Namespace) -> None:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})

        print("Interactive chat (type 'quit' to exit, '/image path' to add image):")
        print("-" * 50)

        while True:
            try:
                user_input = input("User: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break

            if not user_input:
                continue

            # Handle /image command
            if user_input.startswith("/image "):
                image_path = user_input[7:].strip()
                prompt = input("Prompt: ").strip()
                images = [self._load_image(image_path)]
                vision_messages = build_vision_messages(prompt=prompt, images=images)
                messages.extend(vision_messages)
            else:
                messages.append({"role": "user", "content": user_input})

            t0 = time.perf_counter()
            response = client.chat(
                messages=messages,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            latency = time.perf_counter() - t0
            text = response.text
            print(f"Assistant: {text}")
            print(f"  [{latency:.2f}s, {response.usage.completion_tokens} tokens]")

            messages.append({"role": "assistant", "content": text})

    def _cmd_generate(
        self, client: QVLClientsProtocol, args: argparse.Namespace
    ) -> None:
        images = None
        if args.images:
            images = [self._load_image(img) for img in args.images]

        t0 = time.perf_counter()
        result = client.generate(
            prompt=args.prompt,
            images=images,
            system_prompt=args.system_prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        latency = time.perf_counter() - t0
        print(result)
        print(f"\n[{latency:.2f}s]")

    def _cmd_bench(self, client: QVLClientsProtocol, args: argparse.Namespace) -> None:
        n = args.n
        max_tokens = args.max_tokens
        print(f"Running quick benchmark: {n} requests, {max_tokens} max tokens...")

        requests = [
            {"messages": [{"role": "user", "content": f"Count from 1 to {i+1}."}]}
            for i in range(n)
        ]

        t0 = time.perf_counter()
        responses = client.chat_batch(requests, max_tokens=max_tokens, temperature=0.1)
        total_time = time.perf_counter() - t0

        total_tokens = sum(r.usage.total_tokens for r in responses)
        gen_tokens = sum(r.usage.completion_tokens for r in responses)
        prompt_tokens = sum(r.usage.prompt_tokens for r in responses)

        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests/s: {n / total_time:.2f}")
        print(f"  Gen tokens/s: {gen_tokens / total_time:.2f}")
        print(
            f"  Total tokens: {total_tokens} (prompt: {prompt_tokens}, gen: {gen_tokens})"
        )
        print(f"  Avg latency: {total_time / n:.3f}s per request")

    @staticmethod
    def _load_image(path_or_url: str) -> str:
        """Load image as base64 string from file path or return URL."""
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url

        path = Path(path_or_url).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        data = path.read_bytes()
        suffix = path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")

        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"


class QVLClientsArgParser(QVLClientsArgParserBase):
    """Default argument parser that creates QVLClients."""

    def create_client(self, args: argparse.Namespace) -> QVLClientsProtocol:
        from .clients import QVLClients

        return QVLClients(
            endpoints=args.endpoints,
            verbose=args.verbose,
        )


class QVLClientsCLI(QVLClientsCLIBase):
    """Default CLI for QVL clients."""

    def __init__(self):
        super().__init__(QVLClientsArgParser())


def run_cli_main(args: Sequence[str] | None = None) -> None:
    """Entry point for the qvl_clients CLI."""
    cli = QVLClientsCLI()
    cli.run(args)


if __name__ == "__main__":
    run_cli_main()
