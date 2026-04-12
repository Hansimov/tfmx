"""Unified qsr CLI entrypoint."""

import argparse

from . import benchmark, client, compose, machine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qsr",
        description=(
            "Unified CLI for Qwen3-ASR Docker deployment, machine proxying, "
            "client calls, and benchmarking."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  qsr compose up --gpu-layout uniform\n"
            "  qsr machine run --auto-start -b\n"
            "  qsr client transcribe ./sample.wav\n"
            '  qsr client chat --audio ./sample.wav "请转写为简体中文"\n'
            "  qsr benchmark run -E http://localhost:27900 -n 20 --audio ./sample.wav\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compose_parser = subparsers.add_parser(
        "compose", help="Manage Docker compose deployment"
    )
    compose.configure_parser(compose_parser)

    machine_parser = subparsers.add_parser(
        "machine", help="Run and manage the local ASR machine proxy"
    )
    machine.configure_parser(machine_parser)

    client_parser = subparsers.add_parser(
        "client", help="Talk to a machine or direct ASR backend"
    )
    client.configure_parser(client_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark ASR endpoints"
    )
    benchmark.configure_parser(benchmark_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compose":
        if getattr(args, "compose_action", None) is None:
            parser.parse_args(["compose", "--help"])
            return
        compose.run_from_args(args)
    elif args.command == "machine":
        if getattr(args, "action", None) is None:
            parser.parse_args(["machine", "--help"])
            return
        machine.run_from_args(args)
    elif args.command == "client":
        if getattr(args, "client_action", None) is None:
            parser.parse_args(["client", "--help"])
            return
        client.run_from_args(args)
    elif args.command == "benchmark":
        if getattr(args, "benchmark_action", None) is None:
            parser.parse_args(["benchmark", "--help"])
            return
        benchmark.run_from_args(args)
    else:
        raise ValueError(f"Unknown qsr command: {args.command}")


if __name__ == "__main__":
    main()
