"""Benchmark helpers for QSR machine or direct vLLM endpoints."""

import argparse
import itertools
import json
import time

from dataclasses import dataclass, field
from pathlib import Path
from tclogger import logger

from .client import build_audio_messages
from .clients_stats import QSRClientsWithStats


DEFAULT_AUDIO_SAMPLES = [
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
]


@dataclass
class ASRBenchmarkMetrics:
    n_samples: int = 0
    mode: str = "transcribe"
    endpoints: list[str] = field(default_factory=list)
    total_time: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    request_times: list[float] = field(default_factory=list)
    ttft_times: list[float] = field(default_factory=list)
    submitted_requests_per_second: float = 0.0
    requests_per_second: float = 0.0
    total_output_chars: int = 0
    output_chars_per_second: float = 0.0
    total_prompt_tokens: int = 0
    total_gen_tokens: int = 0
    total_tokens: int = 0
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    ttft_min: float = 0.0
    ttft_max: float = 0.0
    ttft_avg: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0

    def calculate_latency_percentiles(self) -> None:
        if not self.request_times:
            return
        sorted_times = sorted(self.request_times)
        n = len(sorted_times)
        self.latency_min = sorted_times[0]
        self.latency_max = sorted_times[-1]
        self.latency_avg = sum(sorted_times) / n
        self.latency_p50 = sorted_times[int(n * 0.50)]
        self.latency_p95 = sorted_times[int(n * 0.95)] if n > 20 else sorted_times[-1]
        self.latency_p99 = sorted_times[int(n * 0.99)] if n > 100 else sorted_times[-1]

    def calculate_ttft_percentiles(self) -> None:
        if not self.ttft_times:
            return
        sorted_times = sorted(self.ttft_times)
        n = len(sorted_times)
        self.ttft_min = sorted_times[0]
        self.ttft_max = sorted_times[-1]
        self.ttft_avg = sum(sorted_times) / n
        self.ttft_p50 = sorted_times[int(n * 0.50)]
        self.ttft_p95 = sorted_times[int(n * 0.95)] if n > 20 else sorted_times[-1]
        self.ttft_p99 = sorted_times[int(n * 0.99)] if n > 100 else sorted_times[-1]

    def to_dict(self) -> dict:
        self.calculate_latency_percentiles()
        self.calculate_ttft_percentiles()
        return {
            "config": {
                "n_samples": self.n_samples,
                "mode": self.mode,
                "endpoints": self.endpoints,
            },
            "requests": {
                "submitted": self.n_samples,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": round(self.success_rate, 4),
            },
            "timing": {
                "total_time_sec": round(self.total_time, 3),
            },
            "throughput": {
                "submitted_requests_per_second": round(
                    self.submitted_requests_per_second, 2
                ),
                "requests_per_second": round(self.requests_per_second, 2),
                "output_chars_per_second": round(self.output_chars_per_second, 2),
            },
            "output": {
                "total_output_chars": self.total_output_chars,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_gen_tokens": self.total_gen_tokens,
                "total_tokens": self.total_tokens,
            },
            "latency_sec": {
                "min": round(self.latency_min, 4),
                "max": round(self.latency_max, 4),
                "avg": round(self.latency_avg, 4),
                "p50": round(self.latency_p50, 4),
                "p95": round(self.latency_p95, 4),
                "p99": round(self.latency_p99, 4),
            },
            "ttft_sec": {
                "min": round(self.ttft_min, 4),
                "max": round(self.ttft_max, 4),
                "avg": round(self.ttft_avg, 4),
                "p50": round(self.ttft_p50, 4),
                "p95": round(self.ttft_p95, 4),
                "p99": round(self.ttft_p99, 4),
            },
        }


class QSRBenchmark:
    def __init__(
        self,
        endpoints: list[str],
        mode: str = "transcribe",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        measure_ttft: bool = True,
        verbose: bool = False,
    ):
        self.endpoints = endpoints
        self.mode = mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.measure_ttft = measure_ttft
        self.verbose = verbose
        self.clients = QSRClientsWithStats(endpoints=endpoints, verbose=verbose)

    def close(self) -> None:
        self.clients.close()

    def __enter__(self) -> "QSRBenchmark":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def check_health(self) -> bool:
        health = self.clients.health()
        logger.mesg(
            f"Healthy machines: {health.healthy_machines}/{health.total_machines}"
        )
        logger.mesg(
            f"Healthy instances: {health.healthy_instances}/{health.total_instances}"
        )
        return health.healthy_machines > 0

    def run(self, requests: list[dict]) -> ASRBenchmarkMetrics:
        metrics = ASRBenchmarkMetrics(
            n_samples=len(requests),
            mode=self.mode,
            endpoints=self.endpoints.copy(),
        )

        started_at = time.perf_counter()
        try:
            if self.mode == "chat":
                outcomes = self.clients.chat_batch_outcomes(
                    requests,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    measure_ttft=self.measure_ttft,
                )
            else:
                outcomes = self.clients.transcribe_batch_outcomes(requests)

            finished_at = time.perf_counter()
            metrics.total_time = finished_at - started_at
            for outcome in outcomes:
                if not outcome.succeeded or outcome.response is None:
                    metrics.failed_requests += 1
                    continue

                metrics.successful_requests += 1
                response = outcome.response
                latency_sec = outcome.latency_sec or getattr(
                    response, "_latency_sec", 0.0
                )
                if latency_sec > 0:
                    metrics.request_times.append(latency_sec)
                if outcome.first_token_latency_sec > 0:
                    metrics.ttft_times.append(outcome.first_token_latency_sec)

                text = getattr(response, "text", "") or ""
                metrics.total_output_chars += len(text)

                usage = getattr(response, "usage", None)
                if usage is not None:
                    metrics.total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                    metrics.total_gen_tokens += getattr(usage, "completion_tokens", 0)
                    metrics.total_tokens += getattr(usage, "total_tokens", 0)
        except Exception as exc:
            metrics.total_time = time.perf_counter() - started_at
            metrics.failed_requests = len(requests)
            logger.warn(f"× Benchmark failed: {exc}")
            return metrics

        if metrics.total_time > 0:
            metrics.submitted_requests_per_second = len(requests) / metrics.total_time
            metrics.requests_per_second = (
                metrics.successful_requests / metrics.total_time
            )
            metrics.output_chars_per_second = (
                metrics.total_output_chars / metrics.total_time
            )

        if metrics.n_samples > 0:
            metrics.success_rate = metrics.successful_requests / metrics.n_samples

        return metrics

    @staticmethod
    def expand_audio_inputs(audio_inputs: list[str], count: int) -> list[str]:
        base_inputs = audio_inputs or DEFAULT_AUDIO_SAMPLES
        return list(itertools.islice(itertools.cycle(base_inputs), count))

    @staticmethod
    def build_chat_requests(
        audio_inputs: list[str],
        prompt: str = "",
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[dict]:
        requests: list[dict] = []
        for audio in audio_inputs:
            requests.append(
                {
                    "messages": build_audio_messages(
                        texts=[prompt] if prompt else [],
                        audios=[audio],
                    ),
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        return requests

    @staticmethod
    def build_transcription_requests(
        audio_inputs: list[str],
        model: str = "",
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
    ) -> list[dict]:
        requests: list[dict] = []
        for audio in audio_inputs:
            requests.append(
                {
                    "audio": audio,
                    "model": model,
                    "language": language,
                    "prompt": prompt,
                    "response_format": response_format,
                    "temperature": temperature,
                }
            )
        return requests


CLI_EPILOG = """
Examples:
  qsr benchmark health -E http://localhost:27900
  qsr benchmark run -E http://localhost:27900 -n 20 --audio ./sample.wav
  qsr benchmark run -E http://localhost:27900 -n 20 --mode chat --audio ./sample.wav --prompt "请转写"
"""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Benchmark QSR machine or direct ASR endpoints"
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = CLI_EPILOG

    subparsers = parser.add_subparsers(dest="benchmark_action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-E", "--endpoints", nargs="+", required=False, default=[])
    common.add_argument("-v", "--verbose", action="store_true")

    health_parser = subparsers.add_parser(
        "health", parents=[common], help="Check endpoint health"
    )
    health_parser.set_defaults(_benchmark_needs_endpoints=True)

    run_parser = subparsers.add_parser("run", parents=[common], help="Run benchmark")
    run_parser.set_defaults(_benchmark_needs_endpoints=True)
    run_parser.add_argument("-n", "--num-samples", type=int, default=20)
    run_parser.add_argument(
        "--mode", choices=["transcribe", "chat"], default="transcribe"
    )
    run_parser.add_argument("--audio", action="append", default=[])
    run_parser.add_argument("--prompt", type=str, default="")
    run_parser.add_argument("--language", type=str, default=None)
    run_parser.add_argument(
        "--response-format",
        choices=["json", "text", "verbose_json", "srt", "vtt"],
        default="json",
    )
    run_parser.add_argument("--model", default="")
    run_parser.add_argument("--max-tokens", type=int, default=512)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument(
        "--no-ttft",
        action="store_true",
        help="Disable streaming-based TTFT measurement for chat mode",
    )
    run_parser.add_argument("-o", "--output", type=str, default=None)


def run_from_args(args: argparse.Namespace) -> None:
    if not args.endpoints:
        raise ValueError("At least one endpoint is required")

    with QSRBenchmark(
        endpoints=args.endpoints,
        mode=getattr(args, "mode", "transcribe"),
        max_tokens=getattr(args, "max_tokens", 512),
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        measure_ttft=not getattr(args, "no_ttft", False),
        verbose=getattr(args, "verbose", False),
    ) as benchmark:
        if args.benchmark_action == "health":
            benchmark.check_health()
            return

        audio_inputs = benchmark.expand_audio_inputs(args.audio, args.num_samples)
        if args.mode == "chat":
            requests = benchmark.build_chat_requests(
                audio_inputs=audio_inputs,
                prompt=args.prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            requests = benchmark.build_transcription_requests(
                audio_inputs=audio_inputs,
                model=args.model,
                language=args.language,
                prompt=args.prompt or None,
                response_format=args.response_format,
                temperature=args.temperature,
            )

        metrics = benchmark.run(requests)
        payload = metrics.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as output_file:
                json.dump(payload, output_file, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
