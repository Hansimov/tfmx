"""Unified qwn CLI entrypoint."""

import argparse

from . import benchmark, client, compose, machine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwn",
        description="Unified CLI for Qwen 3.5 Docker deployment, proxying, debugging, and benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  qwn compose up --gpu-layout uniform-awq\n"
            '  qwn compose up --gpu-configs "0:4b:4bit,1:4b:4bit"\n'
            "  qwn machine run -b\n"
            '  qwn client chat "你好，请做个自我介绍"\n'
            "  qwn benchmark run -E http://localhost:27800 -n 100\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compose_parser = subparsers.add_parser(
        "compose", help="Manage Docker compose deployment"
    )
    compose.configure_parser(compose_parser)

    machine_parser = subparsers.add_parser(
        "machine", help="Run and manage the local proxy"
    )
    machine.configure_parser(machine_parser)

    client_parser = subparsers.add_parser(
        "client", help="Talk to a machine or vLLM endpoint"
    )
    client.configure_parser(client_parser)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark endpoints")
    benchmark.configure_parser(benchmark_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compose":
        compose.run_from_args(args)
    elif args.command == "machine":
        if args.action is None:
            parser.parse_args(["machine", "--help"])
            return
        machine.run_from_args(args)
    elif args.command == "client":
        client.run_from_args(args)
    elif args.command == "benchmark":
        benchmark.run_from_args(args)
    else:
        raise ValueError(f"Unknown qwn command: {args.command}")


if __name__ == "__main__":
    main()
