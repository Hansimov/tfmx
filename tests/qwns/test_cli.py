"""CLI parser tests for qwn."""

from tfmx.qwns.cli import build_parser


class TestQwnCliParser:
    def test_compose_parse(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "up", "--gpu-configs", "0:4b:4bit"])
        assert args.command == "compose"
        assert args.compose_action == "up"

    def test_compose_layout_parse(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "up", "--gpu-layout", "uniform-awq"])
        assert args.command == "compose"
        assert args.gpu_layout == "uniform-awq"

    def test_machine_parse(self):
        parser = build_parser()
        args = parser.parse_args(["machine", "run", "-b"])
        assert args.command == "machine"
        assert args.action == "run"
        assert args.background is True

    def test_client_parse(self):
        parser = build_parser()
        args = parser.parse_args(["client", "chat", "hello"])
        assert args.command == "client"
        assert args.client_action == "chat"
        assert args.text == "hello"

    def test_benchmark_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            ["benchmark", "run", "-E", "http://localhost:27800", "-n", "10"]
        )
        assert args.command == "benchmark"
        assert args.benchmark_action == "run"
        assert args.num_samples == 10
