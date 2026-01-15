"""TEI (Text Embeddings Inference) Benchmark

This module provides benchmarking tools for TEI services,
measuring throughput, latency, and performance across multiple machines.
"""

# ANCHOR[id=benchmark-clis]
CLI_EPILOG = """
Examples:
  # Default benchmark: 200k samples, 2 machines
  export TEI_EPS="http://localhost:28800,http://ai122:28800"
  
  # Note: -E/--endpoints must be placed BEFORE the subcommand
  tei_benchmark -E $TEI_EPS run
  
  # Custom sample count and batch size
  tei_benchmark -E $TEI_EPS run -n 100000     # 100k samples
  tei_benchmark -E $TEI_EPS run -b 3000       # batch size 3000
  
  # Custom text length (longer texts for better GPU utilization)
  tei_benchmark -E $TEI_EPS run --min-len 150 --max-len 400
  
  # Auto-tune batch size for optimal performance
  tei_benchmark -E $TEI_EPS tune              # Test batch sizes 100-2000
  tei_benchmark -E $TEI_EPS tune --min-batch 500 --max-batch 3000 --step 250
  # Note: Scheduler splits large batches into ≤100 per instance automatically
  
  # Benchmark single machine
  tei_benchmark -E "http://localhost:28800" run -n 50000
  
  # Only generate samples (for debugging)
  tei_benchmark generate -n 1000 --show       # Show 10 samples
  
  # LSH bit size options
  tei_benchmark -E $TEI_EPS run --bitn 1024   # 1024 bits
  tei_benchmark -E $TEI_EPS run --bitn 4096   # 4096 bits
  
  # Verbose output and save results
  tei_benchmark -E $TEI_EPS -v run -o results.json
  
  # Health check
  tei_benchmark -E $TEI_EPS health
"""

import argparse
import time
import json

from dataclasses import dataclass, field
from tclogger import logger, logstr

from .tei_clients import TEIClients
from .tei_benchtext import TEIBenchTextGenerator


@dataclass
class BenchmarkMetrics:
    """Metrics collected during a benchmark run."""

    # Configuration
    n_samples: int = 0
    n_batches: int = 0
    batch_size: int = 0
    bitn: int = 2048
    endpoints: list[str] = field(default_factory=list)

    # Timing
    total_time: float = 0.0
    batch_times: list[float] = field(default_factory=list)

    # Throughput
    samples_per_second: float = 0.0
    chars_per_second: float = 0.0

    # Latency (in seconds)
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_avg: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Character stats
    total_chars: int = 0
    avg_chars_per_sample: float = 0.0

    def calculate_latency_percentiles(self) -> None:
        """Calculate latency percentiles from batch times."""
        if not self.batch_times:
            return

        sorted_times = sorted(self.batch_times)
        n = len(sorted_times)

        self.latency_min = sorted_times[0]
        self.latency_max = sorted_times[-1]
        self.latency_avg = sum(sorted_times) / n

        self.latency_p50 = sorted_times[int(n * 0.50)]
        self.latency_p95 = sorted_times[int(n * 0.95)] if n > 20 else sorted_times[-1]
        self.latency_p99 = sorted_times[int(n * 0.99)] if n > 100 else sorted_times[-1]

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "config": {
                "n_samples": self.n_samples,
                "n_batches": self.n_batches,
                "batch_size": self.batch_size,
                "bitn": self.bitn,
                "endpoints": self.endpoints,
            },
            "timing": {
                "total_time_sec": round(self.total_time, 3),
            },
            "throughput": {
                "samples_per_second": round(self.samples_per_second, 2),
                "chars_per_second": round(self.chars_per_second, 2),
            },
            "latency_sec": {
                "min": round(self.latency_min, 4),
                "max": round(self.latency_max, 4),
                "avg": round(self.latency_avg, 4),
                "p50": round(self.latency_p50, 4),
                "p95": round(self.latency_p95, 4),
                "p99": round(self.latency_p99, 4),
            },
            "chars": {
                "total": self.total_chars,
                "avg_per_sample": round(self.avg_chars_per_sample, 1),
            },
        }

    def print_summary(self) -> None:
        """Print a formatted summary of the benchmark results."""
        logger.note("=" * 60)
        logger.note("BENCHMARK RESULTS")
        logger.note("=" * 60)

        # Configuration
        logger.mesg(f"\n[Configuration]")
        logger.mesg(f"  Samples:    {self.n_samples:,}")
        # logger.mesg(f"  Batches:    {self.n_batches:,}")
        # logger.mesg(f"  Batch size: {self.batch_size:,}")
        logger.mesg(f"  LSH bits:   {self.bitn}")
        logger.mesg(f"  Endpoints:  {len(self.endpoints)}")
        for ep in self.endpoints:
            logger.mesg(f"  - {ep}")

        # Timing
        logger.mesg(f"\n[Timing]")
        logger.mesg(f"  Total time: {self.total_time:.2f} sec")

        # Throughput
        logger.mesg(f"\n[Throughput]")
        logger.mesg(f"  Samples/sec: {logstr.mesg(f'{self.samples_per_second:,.0f}')}")


class TEIBenchmark:
    """Benchmark runner for TEI services.

    Measures LSH performance across multiple machines with detailed metrics.
    """

    def __init__(
        self,
        endpoints: list[str],
        batch_size: int = 1000,
        bitn: int = 2048,
        timeout: float = 120.0,
        verbose: bool = False,
        skip_exploration: bool = False,
    ):
        """Initialize the benchmark runner.

        Args:
            endpoints: List of TEI machine endpoints
            batch_size: Number of samples per batch
            bitn: Number of LSH bits
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            skip_exploration: If True, use saved optimal batch sizes instead of re-exploring
        """
        self.endpoints = endpoints
        self.batch_size = batch_size
        self.bitn = bitn
        self.timeout = timeout
        self.verbose = verbose
        self.skip_exploration = skip_exploration

        # Initialize client
        self.clients = TEIClients(
            endpoints=endpoints,
            timeout=timeout,
            verbose=verbose,
            skip_exploration=skip_exploration,
        )

    def close(self) -> None:
        """Close the clients."""
        self.clients.close()

    def __enter__(self) -> "TEIBenchmark":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def check_health(self) -> bool:
        """Check if all endpoints are healthy.

        Returns:
            True if at least one endpoint is healthy
        """
        logger.note("> Checking endpoint health...")
        health = self.clients.health()

        logger.mesg(
            f"  Healthy machines:   {health.healthy_machines}/{health.total_machines}"
        )
        logger.mesg(
            f"  Healthy instances:  {health.healthy_instances}/{health.total_instances}"
        )

        if health.healthy_machines == 0:
            logger.warn("× No healthy machines available!")
            return False

        logger.okay(f"  Status: {health.status}")
        return True

    def run(
        self,
        samples: list[str],
        warmup_samples: int = 100,
    ) -> BenchmarkMetrics:
        """Run the benchmark using pipeline mode.

        Pipeline mode: All samples are processed together, with the scheduler
        distributing work across machines. Each machine uses its optimal batch size.

        Args:
            samples: List of text samples to process
            warmup_samples: Number of samples for warmup (default: 100)

        Returns:
            BenchmarkMetrics with detailed results
        """
        # Initialize metrics
        metrics = BenchmarkMetrics(
            n_samples=len(samples),
            batch_size=self.batch_size,  # Note: this is informational only in pipeline mode
            bitn=self.bitn,
            endpoints=self.endpoints.copy(),
        )

        # Calculate character stats
        metrics.total_chars = sum(len(s) for s in samples)
        metrics.avg_chars_per_sample = (
            metrics.total_chars / len(samples) if samples else 0
        )

        # Warmup
        if warmup_samples > 0:
            logger.note(f"> Warmup with {warmup_samples} samples...")
            warmup_data = samples[:warmup_samples]
            try:
                self.clients.lsh(warmup_data, bitn=self.bitn)
                logger.okay("  Warmup completed")
            except Exception as e:
                logger.warn(f"  Warmup failed: {e}")

        logger.note(f"> Running benchmark (pipeline mode)...")
        logger.mesg(f"  Samples: {len(samples):,}")
        logger.mesg(f"  Endpoints: {len(self.endpoints)}")

        # Create a generator that yields samples one by one
        def sample_generator():
            for sample in samples:
                yield sample

        # Run benchmark - use lsh_iter with generator, provide total_hint for progress
        start_time = time.perf_counter()

        try:
            results = self.clients.lsh_iter(
                sample_generator(),
                total_hint=len(samples),
                bitn=self.bitn,
            )
            end_time = time.perf_counter()
            total_processed = len(results)
        except Exception as e:
            logger.error(f"  Benchmark failed: {e}")
            end_time = time.perf_counter()
            total_processed = 0

        # Calculate metrics
        metrics.total_time = end_time - start_time
        metrics.n_batches = 1  # Pipeline mode = 1 logical batch

        if metrics.total_time > 0:
            metrics.samples_per_second = total_processed / metrics.total_time
            metrics.chars_per_second = metrics.total_chars / metrics.total_time

        # Get per-machine stats with elapsed time for accurate throughput calculation
        logger.note(f"Machine stats:")
        machine_stats = self.clients.get_machine_stats(elapsed_time=metrics.total_time)
        for ms in machine_stats:
            logger.file(
                f"  {ms['endpoint']:<22}: {ms['total_items']:,} items, "
                f"{ms['total_requests']:>2} batches, "
                f"throughput={ms['throughput']:.0f}/s"
            )

        logger.okay(f"  Benchmark completed in {metrics.total_time:.2f} sec")

        return metrics


class TEIBenchmarkArgParser:
    """Argument parser for TEI Benchmark CLI."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="TEI Benchmark - Performance testing for TEI services",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=CLI_EPILOG,
        )
        self._setup_arguments()
        self.args = self.parser.parse_args()

    def _setup_arguments(self):
        """Setup all command-line arguments."""
        # Common arguments
        self.parser.add_argument(
            "-E",
            "--endpoints",
            type=str,
            default=None,
            help="Comma-separated list of TEI machine endpoints (e.g., http://localhost:28800,http://ai122:28800)",
        )
        self.parser.add_argument(
            "-n",
            "--num-samples",
            type=int,
            default=100000,
            help="Number of text samples to generate (default: 100000)",
        )
        self.parser.add_argument(
            "-b",
            "--batch-size",
            type=int,
            default=2000,
            help="Batch size for requests (default: 2000, scheduler splits into ≤100 per instance)",
        )
        self.parser.add_argument(
            "--bitn",
            type=int,
            default=2048,
            help="Number of LSH bits (default: 2048)",
        )
        self.parser.add_argument(
            "--min-len",
            type=int,
            default=100,
            help="Minimum text length in characters (default: 100)",
        )
        self.parser.add_argument(
            "--max-len",
            type=int,
            default=300,
            help="Maximum text length in characters (default: 300)",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for text generation (default: 42)",
        )
        self.parser.add_argument(
            "--warmup",
            type=int,
            default=100,
            help="Number of warmup samples (default: 100)",
        )
        self.parser.add_argument(
            "-t",
            "--timeout",
            type=float,
            default=120.0,
            help="Request timeout in seconds (default: 120.0)",
        )
        self.parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )
        self.parser.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help="Output file for results (JSON format)",
        )
        self.parser.add_argument(
            "--explore",
            action="store_true",
            default=False,
            help="Force re-exploration of optimal batch sizes (ignore saved config)",
        )

        # Action subcommands
        subparsers = self.parser.add_subparsers(dest="action", help="Action to perform")

        # run - main benchmark
        run_parser = subparsers.add_parser("run", help="Run the benchmark")

        # tune - auto-tune batch size
        tune_parser = subparsers.add_parser(
            "tune", help="Auto-tune batch size for optimal performance"
        )
        tune_parser.add_argument(
            "--min-batch",
            type=int,
            default=100,
            help="Minimum batch size to test (default: 100, scheduler will split into micro-batches of ≤100 per instance)",
        )
        tune_parser.add_argument(
            "--max-batch",
            type=int,
            default=2000,
            help="Maximum batch size to test (default: 2000, large batches utilize multiple instances in parallel)",
        )
        tune_parser.add_argument(
            "--step",
            type=int,
            default=100,
            help="Batch size increment step (default: 100)",
        )
        tune_parser.add_argument(
            "--test-samples",
            type=int,
            default=50000,
            help="Number of samples for each test (default: 50000, enough to utilize multiple GPU instances)",
        )

        # generate - only generate samples
        gen_parser = subparsers.add_parser(
            "generate", help="Only generate text samples"
        )
        gen_parser.add_argument(
            "--show",
            action="store_true",
            help="Show sample texts",
        )
        gen_parser.add_argument(
            "--show-count",
            type=int,
            default=10,
            help="Number of samples to show (default: 10)",
        )

        # health - check endpoint health
        health_parser = subparsers.add_parser("health", help="Check endpoint health")


def main():
    """Main entry point for CLI."""
    arg_parser = TEIBenchmarkArgParser()
    args = arg_parser.args

    if args.action is None:
        arg_parser.parser.print_help()
        return

    # Generate action
    if args.action == "generate":
        generator = TEIBenchTextGenerator(seed=args.seed)
        samples = generator.generate(
            count=args.num_samples,
            min_len=args.min_len,
            max_len=args.max_len,
        )

        if args.show:
            logger.note(f"\n> Sample texts (first {args.show_count}):")
            for i, sample in enumerate(samples[: args.show_count]):
                logger.mesg(f"  [{i + 1}] ({len(sample)} chars): {sample[:100]}...")
        return

    # Health check
    if args.action == "health":
        if not args.endpoints:
            logger.warn("× No endpoints specified. Use -E to specify endpoints.")
            return

        endpoints = [ep.strip() for ep in args.endpoints.split(",")]
        clients = TEIClients(endpoints=endpoints, verbose=args.verbose)
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

    # Tune batch size
    if args.action == "tune":
        if not args.endpoints:
            logger.warn("× No endpoints specified. Use -E to specify endpoints.")
            return

        endpoints = [ep.strip() for ep in args.endpoints.split(",")]

        # Generate test samples
        logger.note(f"> Auto-tuning batch size...")
        generator = TEIBenchTextGenerator(seed=args.seed)
        samples = generator.generate(
            count=args.test_samples,
            min_len=args.min_len,
            max_len=args.max_len,
        )

        # Test different batch sizes
        batch_sizes = list(range(args.min_batch, args.max_batch + 1, args.step))
        results = []

        logger.note(
            f"\n> Testing {len(batch_sizes)} batch sizes: {batch_sizes[0]} to {batch_sizes[-1]}"
        )

        for batch_size in batch_sizes:
            logger.mesg(f"\n  Testing batch_size={batch_size}...")

            with TEIBenchmark(
                endpoints=endpoints,
                batch_size=batch_size,
                bitn=args.bitn,
                timeout=args.timeout,
                verbose=False,  # Disable verbose for cleaner output
            ) as benchmark:
                if not benchmark.check_health():
                    logger.warn("× Health check failed, skipping tune")
                    return

                try:
                    metrics = benchmark.run(
                        samples=samples,
                        warmup_samples=min(100, args.test_samples // 10),
                    )

                    results.append(
                        {
                            "batch_size": batch_size,
                            "throughput": metrics.samples_per_second,
                            "latency_avg": metrics.latency_avg,
                            "latency_p95": metrics.latency_p95,
                        }
                    )

                    logger.okay(
                        f"    Throughput: {metrics.samples_per_second:,.1f} samples/sec, "
                        f"P95: {metrics.latency_p95 * 1000:.1f} ms"
                    )
                except Exception as e:
                    error_msg = str(e)
                    if (
                        "maximum allowed batch size" in error_msg
                        or "Validation" in error_msg
                    ):
                        logger.warn(
                            f"    Batch size {batch_size} exceeds TEI container limit. "
                            f"Stopping tune at batch_size={batch_size - args.step}"
                        )
                        break
                    else:
                        logger.warn(f"    Batch size {batch_size} failed: {e}")
                        # Continue testing other batch sizes
                        continue

        # Find optimal batch size
        best = max(results, key=lambda x: x["throughput"])

        logger.note(f"\n{'=' * 60}")
        logger.note("BATCH SIZE TUNING RESULTS")
        logger.note(f"{'=' * 60}")
        logger.mesg(f"\n  Optimal batch size: {logstr.okay(str(best['batch_size']))}")
        logger.mesg(f"  Peak throughput:    {best['throughput']:,.1f} samples/sec")
        logger.mesg(f"  Avg latency:        {best['latency_avg'] * 1000:.1f} ms")
        logger.mesg(f"  P95 latency:        {best['latency_p95'] * 1000:.1f} ms")

        logger.note(f"\n  All results:")
        for r in results:
            marker = " ← BEST" if r == best else ""
            logger.mesg(
                f"    batch={r['batch_size']:4d}: "
                f"{r['throughput']:7.1f} samples/sec, "
                f"P95={r['latency_p95'] * 1000:6.1f} ms{marker}"
            )
        logger.note(f"{'=' * 60}")
        return

    # Run benchmark
    if args.action == "run":
        if not args.endpoints:
            logger.warn("× No endpoints specified. Use -E to specify endpoints.")
            return

        endpoints = [ep.strip() for ep in args.endpoints.split(",")]

        # Generate samples
        generator = TEIBenchTextGenerator(seed=args.seed)
        samples = generator.generate(
            count=args.num_samples,
            min_len=args.min_len,
            max_len=args.max_len,
        )

        # Run benchmark
        # Default: use saved config (skip_exploration=True)
        # With --explore: force re-exploration (skip_exploration=False)
        skip_exploration = not args.explore

        with TEIBenchmark(
            endpoints=endpoints,
            batch_size=args.batch_size,
            bitn=args.bitn,
            timeout=args.timeout,
            verbose=args.verbose,
            skip_exploration=skip_exploration,
        ) as benchmark:
            # Check health first
            if not benchmark.check_health():
                return

            # Run benchmark
            metrics = benchmark.run(
                samples=samples,
                warmup_samples=args.warmup,
            )

            # Print results
            metrics.print_summary()

            # Save results if output specified
            if args.output:
                results = metrics.to_dict()
                results["scheduler_stats"] = stats
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.okay(f"\n> Results saved to: {args.output}")


if __name__ == "__main__":
    main()

    # LINK: src/tfmx/tei_benchmark.py#benchmark-clis
