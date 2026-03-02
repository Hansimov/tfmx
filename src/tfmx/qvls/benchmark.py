"""QVL Benchmark - Performance Testing for QVL (Qwen3-VL) Services

Benchmarking tool for measuring throughput and performance of QVL
chat completion services across multiple machines.

Features:
- Multi-machine load testing with synthetic images + prompts
- Real-time progress tracking
- Per-machine throughput monitoring
- Tokens-per-second and latency metrics
- JSON results export

Usage:
    # Basic benchmark
    qvl_benchmark run -E "http://m1:29800,http://m2:29800" -n 100

    # Text-only benchmark
    qvl_benchmark run -E "http://m1:29800" -n 50 --text-only

    # Verbose with results saved
    qvl_benchmark run -E "http://m1:29800" -v -o results.json

    # Health check
    qvl_benchmark health -E "http://m1:29800"
"""

# ANCHOR[id=qvl-benchmark-clis]
CLI_EPILOG = """
Examples:
  export QVL_EPS="http://localhost:29800,http://ai122:29800"

  # Basic benchmark
  qvl_benchmark run -E $QVL_EPS -n 100

  # Text-only (no images)
  qvl_benchmark run -E $QVL_EPS -n 200 --text-only

  # Custom max tokens
  qvl_benchmark run -E $QVL_EPS -n 100 --max-tokens 256

  # Verbose with output
  qvl_benchmark run -E $QVL_EPS -v -o results.json

  # Health check
  qvl_benchmark health -E $QVL_EPS

  # Generate sample prompts only
  qvl_benchmark generate -n 20 --show
"""

import argparse
import json
import time

from dataclasses import dataclass, field
from tclogger import logger, logstr

from .clients_stats import QVLClientsWithStats
from .benchimgs import QVLBenchImageGenerator


@dataclass
class BenchmarkMetrics:
    """Metrics collected during a benchmark run."""

    # Configuration
    n_samples: int = 0
    max_tokens: int = 512
    temperature: float = 0.1
    text_only: bool = False
    endpoints: list[str] = field(default_factory=list)

    # Timing
    total_time: float = 0.0
    request_times: list[float] = field(default_factory=list)

    # Throughput
    requests_per_second: float = 0.0
    gen_tokens_per_second: float = 0.0
    total_tokens_per_second: float = 0.0

    # Token stats
    total_prompt_tokens: int = 0
    total_gen_tokens: int = 0
    total_tokens: int = 0

    # Latency (in seconds)
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

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

    def to_dict(self) -> dict:
        return {
            "config": {
                "n_samples": self.n_samples,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "text_only": self.text_only,
                "endpoints": self.endpoints,
            },
            "timing": {
                "total_time_sec": round(self.total_time, 3),
            },
            "throughput": {
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
        }


class QVLBenchmark:
    """Benchmark runner for QVL chat completion services.

    Measures throughput (tokens/s), latency, and request rates
    across multiple machines.
    """

    def __init__(
        self,
        endpoints: list[str],
        max_tokens: int = 128,
        temperature: float = 0.1,
        verbose: bool = False,
    ):
        self.endpoints = endpoints
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

        self.clients = QVLClientsWithStats(
            endpoints=endpoints,
            verbose=verbose,
        )

        logger.note("> Loaded configuration:")
        logger.mesg(f"  Endpoints: {len(self.endpoints)}")
        logger.mesg(f"  Max tokens: {self.max_tokens}")

    def close(self) -> None:
        self.clients.close()

    def __enter__(self) -> "QVLBenchmark":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def check_health(self) -> bool:
        logger.note("> Checking endpoint health...")
        health = self.clients.health()
        logger.mesg(
            f"  Healthy machines:  {health.healthy_machines}/{health.total_machines}"
        )
        logger.mesg(
            f"  Healthy instances: {health.healthy_instances}/{health.total_instances}"
        )

        if health.healthy_machines == 0:
            logger.warn("× No healthy machines available!")
            return False

        logger.okay(f"  Status: {health.status}")
        return True

    def run(
        self,
        requests: list[dict],
        text_only: bool = False,
    ) -> BenchmarkMetrics:
        """Run the benchmark.

        Args:
            requests: List of chat request dicts with 'messages' key
            text_only: Whether requests are text-only (no images)

        Returns:
            BenchmarkMetrics with detailed results
        """
        metrics = BenchmarkMetrics(
            n_samples=len(requests),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            text_only=text_only,
            endpoints=self.endpoints.copy(),
        )

        logger.note(f"> Running benchmark...")
        logger.mesg(f"  Requests: {len(requests):,}")
        logger.mesg(f"  Max tokens: {self.max_tokens}")
        logger.mesg(f"  Endpoints: {len(self.endpoints)}")

        start_time = time.perf_counter()

        try:
            responses = self.clients.chat_batch(
                requests,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            end_time = time.perf_counter()

            # Collect token stats
            for resp in responses:
                metrics.total_prompt_tokens += resp.usage.prompt_tokens
                metrics.total_gen_tokens += resp.usage.completion_tokens
                metrics.total_tokens += resp.usage.total_tokens

        except Exception as e:
            logger.error(f"  Benchmark failed: {e}")
            end_time = time.perf_counter()
            responses = []

        metrics.total_time = end_time - start_time

        if metrics.total_time > 0 and responses:
            metrics.requests_per_second = len(responses) / metrics.total_time
            metrics.gen_tokens_per_second = (
                metrics.total_gen_tokens / metrics.total_time
            )
            metrics.total_tokens_per_second = metrics.total_tokens / metrics.total_time

        # Log summary
        total_time_str = logstr.okay(f"{metrics.total_time:.1f}s")
        tokens_str = logstr.okay(f"{metrics.gen_tokens_per_second:.0f} tok/s")
        rps_str = logstr.mesg(f"{metrics.requests_per_second:.1f} req/s")

        logger.okay(
            f"Benchmark completed in {total_time_str}: " f"{tokens_str}, {rps_str}"
        )
        logger.mesg(
            f"  Tokens: {metrics.total_gen_tokens} generated, "
            f"{metrics.total_prompt_tokens} prompt, "
            f"{metrics.total_tokens} total"
        )

        return metrics


class QVLBenchmarkArgParser:
    """Argument parser for QVL Benchmark CLI."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="QVL Benchmark - Performance testing for QVL services",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        subparsers = self.parser.add_subparsers(
            dest="action",
            help="Action to perform",
            required=True,
        )

        # Common parent parser
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser.add_argument(
            "-E",
            "--endpoints",
            type=str,
            default=None,
            help="Comma-separated list of QVL machine endpoints",
        )
        parent_parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )
        parent_parser.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help="Output file for results (JSON)",
        )

        # run - main benchmark
        run_parser = subparsers.add_parser(
            "run",
            help="Run the benchmark",
            parents=[parent_parser],
        )
        run_parser.add_argument(
            "-n",
            "--num-samples",
            type=int,
            default=100,
            help="Number of requests (default: 100)",
        )
        run_parser.add_argument(
            "--max-tokens",
            type=int,
            default=128,
            help="Max tokens per request (default: 128)",
        )
        run_parser.add_argument(
            "--temperature",
            type=float,
            default=0.1,
            help="Temperature (default: 0.1)",
        )
        run_parser.add_argument(
            "--text-only",
            action="store_true",
            help="Use text-only prompts (no images)",
        )
        run_parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed (default: 42)",
        )
        run_parser.add_argument(
            "--img-size",
            type=int,
            default=256,
            help="Synthetic image size in pixels (default: 256)",
        )

        # generate - only generate samples
        gen_parser = subparsers.add_parser(
            "generate",
            help="Only generate test prompts",
        )
        gen_parser.add_argument(
            "-n",
            "--num-samples",
            type=int,
            default=20,
            help="Number of prompts to generate (default: 20)",
        )
        gen_parser.add_argument(
            "--text-only",
            action="store_true",
            help="Generate text-only prompts",
        )
        gen_parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed (default: 42)",
        )
        gen_parser.add_argument(
            "--show",
            action="store_true",
            help="Show generated prompts",
        )
        gen_parser.add_argument(
            "--show-count",
            type=int,
            default=10,
            help="Number of prompts to show (default: 10)",
        )

        # health - check endpoint health
        subparsers.add_parser(
            "health",
            help="Check endpoint health",
            parents=[parent_parser],
        )


def main():
    """Main entry point for CLI."""
    arg_parser = QVLBenchmarkArgParser()
    args = arg_parser.args

    if args.action is None:
        arg_parser.parser.print_help()
        return

    # Generate action
    if args.action == "generate":
        generator = QVLBenchImageGenerator(seed=args.seed)

        if args.text_only:
            samples = generator.generate_text_only(count=args.num_samples)
        else:
            samples = generator.generate(count=args.num_samples)

        if args.show:
            logger.note(f"\n> Sample prompts (first {args.show_count}):")
            for i, sample in enumerate(samples[: args.show_count]):
                msgs = sample["messages"]
                user_msg = next((m for m in msgs if m["role"] == "user"), None)
                if user_msg:
                    content = user_msg["content"]
                    if isinstance(content, str):
                        logger.mesg(f"  [{i + 1}] text: {content[:80]}...")
                    elif isinstance(content, list):
                        text_parts = [
                            p["text"]
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        img_parts = [
                            p
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "image_url"
                        ]
                        logger.mesg(
                            f"  [{i + 1}] text: {text_parts[0][:60] if text_parts else '?'}... "
                            f"({len(img_parts)} image(s))"
                        )
        return

    # Health check
    if args.action == "health":
        if not args.endpoints:
            logger.warn("× No endpoints specified. Use -E to specify endpoints.")
            return

        endpoints = [ep.strip() for ep in args.endpoints.split(",")]
        clients = QVLClientsWithStats(endpoints=endpoints, verbose=args.verbose)
        try:
            health = clients.health()
            logger.note(f"> Health check results:")
            logger.mesg(f"  Status: {health.status}")
            logger.mesg(
                f"  Healthy machines:  {health.healthy_machines}/{health.total_machines}"
            )
            logger.mesg(
                f"  Healthy instances: {health.healthy_instances}/{health.total_instances}"
            )
        finally:
            clients.close()
        return

    # Run benchmark
    if args.action == "run":
        if not args.endpoints:
            logger.warn("× No endpoints specified. Use -E to specify endpoints.")
            return

        endpoints = [ep.strip() for ep in args.endpoints.split(",")]

        # Generate test requests
        generator = QVLBenchImageGenerator(seed=args.seed)

        if args.text_only:
            requests = generator.generate_text_only(count=args.num_samples)
        else:
            requests = generator.generate(
                count=args.num_samples,
                img_size=args.img_size,
            )

        # Run benchmark
        with QVLBenchmark(
            endpoints=endpoints,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
        ) as benchmark:
            if not benchmark.check_health():
                return

            metrics = benchmark.run(
                requests=requests,
                text_only=args.text_only,
            )

            if args.output:
                results = metrics.to_dict()
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.okay(f"\n> Results saved to: {args.output}")


if __name__ == "__main__":
    main()
