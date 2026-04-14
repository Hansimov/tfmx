"""CLI parser tests for tei."""

from tfmx.teis.cli import build_parser


class TestTeiCliParser:
    def test_compose_parse(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "up", "-g", "0,1"])
        assert args.command == "compose"
        assert args.compose_action == "up"
        assert args.gpus == "0,1"

    def test_compose_parse_gpu_configs(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "up", "--gpu-configs", "0,1"])
        assert args.command == "compose"
        assert args.compose_action == "up"
        assert args.gpu_configs == "0,1"

    def test_machine_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "machine",
                "run",
                "--auto-start",
                "--compose-gpus",
                "0,2",
                "--compose-gpu-configs",
                "0,2",
            ]
        )
        assert args.command == "machine"
        assert args.action == "run"
        assert args.auto_start is True
        assert args.compose_gpus == "0,2"
        assert args.compose_gpu_configs == "0,2"

    def test_machine_parse_background(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "machine",
                "run",
                "--background",
                "--auto-start",
                "--compose-gpus",
                "0,1,2,3,4,5",
            ]
        )
        assert args.command == "machine"
        assert args.action == "run"
        assert args.background is True
        assert args.auto_start is True
        assert args.compose_gpus == "0,1,2,3,4,5"

    def test_machine_parse_on_conflict(self):
        parser = build_parser()
        args = parser.parse_args(["machine", "run", "--on-conflict", "replace"])
        assert args.command == "machine"
        assert args.action == "run"
        assert args.on_conflict == "replace"

    def test_machine_parse_status(self):
        parser = build_parser()
        args = parser.parse_args(["machine", "status"])
        assert args.command == "machine"
        assert args.action == "status"

    def test_client_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            ["client", "embed", "-E", "http://localhost:28800", "hello"]
        )
        assert args.command == "client"
        assert args.client_action == "embed"
        assert args.endpoints == "http://localhost:28800"
        assert args.texts == ["hello"]

    def test_benchmark_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            ["benchmark", "run", "-E", "http://localhost:28800", "-n", "10"]
        )
        assert args.command == "benchmark"
        assert args.benchmark_action == "run"
        assert args.num_samples == 10
