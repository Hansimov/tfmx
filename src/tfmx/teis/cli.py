"""Unified tei CLI entrypoint."""

import argparse

from . import benchmark, client, compose, machine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tei",
        description=(
            "Unified CLI for TEI docker deployment, machine proxying, client calls, "
            "and benchmarking."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  tei compose up\n"
            "  tei machine run --auto-start\n"
            '  tei client -E "$TEI_MACHINE_A_URL,$TEI_MACHINE_B_URL" lsh "Hello"\n'
            '  tei benchmark run -E "$TEI_MACHINE_A_URL" -n 10000\n'
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compose_parser = subparsers.add_parser(
        "compose", help="Manage Docker compose deployment"
    )
    compose.configure_parser(compose_parser)

    machine_parser = subparsers.add_parser(
        "machine", help="Run and manage the local TEI machine proxy"
    )
    machine.configure_parser(machine_parser)

    client_parser = subparsers.add_parser(
        "client", help="Talk to a TEI machine, container, or machine cluster"
    )
    client.configure_parser(client_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark TEI machine endpoints"
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
        raise ValueError(f"Unknown tei command: {args.command}")


if __name__ == "__main__":
    main()
