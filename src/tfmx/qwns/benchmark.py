"""Benchmark helpers for QWN machine or direct vLLM endpoints."""

import argparse
import json
import time

from dataclasses import dataclass, field
from pathlib import Path
from tclogger import logger

from tfmx.teis.benchtext import TEIBenchTextGenerator

from .client import build_text_messages
from .clients_stats import QWNClientsWithStats


@dataclass
class BenchmarkMetrics:
    n_samples: int = 0
    max_tokens: int = 512
    temperature: float = 0.1
    endpoints: list[str] = field(default_factory=list)
    total_time: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    request_times: list[float] = field(default_factory=list)
    ttft_times: list[float] = field(default_factory=list)
    submitted_requests_per_second: float = 0.0
    requests_per_second: float = 0.0
    gen_tokens_per_second: float = 0.0
    total_tokens_per_second: float = 0.0
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
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
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
                "gen_tokens_per_second": round(self.gen_tokens_per_second, 2),
                "total_tokens_per_second": round(self.total_tokens_per_second, 2),
            },
            "tokens": {
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


class QWNBenchmark:
    def __init__(
        self,
        endpoints: list[str],
        max_tokens: int = 128,
        temperature: float = 0.1,
        measure_ttft: bool = True,
        verbose: bool = False,
    ):
        self.endpoints = endpoints
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.measure_ttft = measure_ttft
        self.verbose = verbose
        self.clients = QWNClientsWithStats(endpoints=endpoints, verbose=verbose)

    def close(self) -> None:
        self.clients.close()

    def __enter__(self) -> "QWNBenchmark":
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

    def run(self, requests: list[dict]) -> BenchmarkMetrics:
        metrics = BenchmarkMetrics(
            n_samples=len(requests),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            endpoints=self.endpoints.copy(),
        )

        started_at = time.perf_counter()
        try:
            outcomes = self.clients.chat_batch_outcomes(
                requests,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                measure_ttft=self.measure_ttft,
            )
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
                metrics.total_prompt_tokens += response.usage.prompt_tokens
                metrics.total_gen_tokens += response.usage.completion_tokens
                metrics.total_tokens += response.usage.total_tokens
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
            metrics.gen_tokens_per_second = (
                metrics.total_gen_tokens / metrics.total_time
            )
            metrics.total_tokens_per_second = metrics.total_tokens / metrics.total_time

        if metrics.n_samples > 0:
            metrics.success_rate = metrics.successful_requests / metrics.n_samples

        return metrics

    @staticmethod
    def build_requests(
        prompts: list[str],
        model: str = "",
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> list[dict]:
        requests: list[dict] = []
        for prompt in prompts:
            requests.append(
                {
                    "messages": build_text_messages(prompt=prompt),
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
        return requests

    @staticmethod
    def generate_prompts(
        count: int,
        min_len: int = 60,
        max_len: int = 180,
        seed: int = 42,
        show_progress: bool = False,
    ) -> list[str]:
        generator = TEIBenchTextGenerator(seed=seed, num_workers=1)
        return generator.generate(
            count=count,
            min_len=min_len,
            max_len=max_len,
            show_progress=show_progress,
        )


CLI_EPILOG = """
Examples:
  qwn benchmark health -E http://localhost:27800
  qwn benchmark run -E http://localhost:27800 -n 100
  qwn benchmark run -E http://localhost:27800 http://host2:27800 -n 100 -o result.json
  qwn benchmark generate -n 5 --show
"""


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "Benchmark QWN machine or direct vLLM endpoints"
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
    run_parser.add_argument("-n", "--num-samples", type=int, default=100)
    run_parser.add_argument("--min-len", type=int, default=60)
    run_parser.add_argument("--max-len", type=int, default=180)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--model", default="")
    run_parser.add_argument("--max-tokens", type=int, default=128)
    run_parser.add_argument("--temperature", type=float, default=0.1)
    run_parser.add_argument(
        "--no-ttft",
        action="store_true",
        help="Disable streaming-based TTFT measurement",
    )
    run_parser.add_argument("-o", "--output", type=str, default=None)

    generate_parser = subparsers.add_parser("generate", help="Generate sample prompts")
    generate_parser.add_argument("-n", "--num-samples", type=int, default=10)
    generate_parser.add_argument("--min-len", type=int, default=60)
    generate_parser.add_argument("--max-len", type=int, default=180)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--show", action="store_true")


def run_from_args(args: argparse.Namespace) -> None:
    if args.benchmark_action == "generate":
        prompts = QWNBenchmark.generate_prompts(
            count=args.num_samples,
            min_len=args.min_len,
            max_len=args.max_len,
            seed=args.seed,
            show_progress=False,
        )
        if args.show:
            for prompt in prompts:
                print(prompt)
        else:
            print(json.dumps(prompts, ensure_ascii=False, indent=2))
        return

    if not args.endpoints:
        raise ValueError("At least one endpoint is required")

    with QWNBenchmark(
        endpoints=args.endpoints,
        max_tokens=getattr(args, "max_tokens", 128),
        temperature=getattr(args, "temperature", 0.1),
        measure_ttft=not getattr(args, "no_ttft", False),
        verbose=getattr(args, "verbose", False),
    ) as benchmark:
        if args.benchmark_action == "health":
            benchmark.check_health()
            return

        prompts = benchmark.generate_prompts(
            count=args.num_samples,
            min_len=args.min_len,
            max_len=args.max_len,
            seed=args.seed,
            show_progress=False,
        )
        requests = benchmark.build_requests(
            prompts=prompts,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        metrics = benchmark.run(requests)
        metrics.calculate_latency_percentiles()
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
