"""CLI parser tests for qsr."""

from tfmx.qsrs.cli import build_parser


class TestQsrCliParser:
    def test_compose_up_default_warmup_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "compose",
                "up",
                "--warmup-audio",
                "./sample.wav",
                "--no-skip-mm-profiling",
                "--cudagraph-mode",
                "FULL",
                "--poll-interval",
                "0.5",
            ]
        )
        assert args.command == "compose"
        assert args.compose_action == "up"
        assert args.skip_warmup is False
        assert args.warmup_audio == "./sample.wav"
        assert args.skip_mm_profiling is False
        assert args.cudagraph_mode == "FULL"
        assert args.poll_interval == 0.5

    def test_compose_up_startup_profile_parse(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "up", "--profile-startup"])

        assert args.command == "compose"
        assert args.compose_action == "up"
        assert args.profile_startup is True

    def test_compose_warmup_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "compose",
                "warmup",
                "--audio",
                "./sample.wav",
                "--wait-timeout",
                "30",
            ]
        )
        assert args.command == "compose"
        assert args.compose_action == "warmup"
        assert args.audio == "./sample.wav"
        assert args.wait_timeout == 30.0

    def test_compose_sleep_status_parse(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "sleep-status", "--gpu-layout", "uniform"])

        assert args.command == "compose"
        assert args.compose_action == "sleep-status"
        assert args.gpu_layout == "uniform"

    def test_compose_wake_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "compose",
                "up",
                "--enable-sleep-mode",
            ]
        )

        assert args.command == "compose"
        assert args.compose_action == "up"
        assert args.enable_sleep_mode is True

    def test_compose_parse_gpu_configs(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "up", "--gpu-configs", "0,1"])
        assert args.command == "compose"
        assert args.compose_action == "up"
        assert args.gpu_configs == "0,1"

    def test_compose_layout_parse(self):
        parser = build_parser()
        args = parser.parse_args(["compose", "up", "--gpu-layout", "uniform"])
        assert args.command == "compose"
        assert args.gpu_layout == "uniform"

    def test_machine_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "machine",
                "run",
                "--auto-start",
                "--compose-gpus",
                "0",
                "-b",
            ]
        )
        assert args.command == "machine"
        assert args.action == "run"
        assert args.auto_start is True
        assert args.compose_gpus == "0"
        assert args.background is True

    def test_machine_parse_on_conflict(self):
        parser = build_parser()
        args = parser.parse_args(["machine", "run", "--on-conflict", "replace"])
        assert args.command == "machine"
        assert args.action == "run"
        assert args.on_conflict == "replace"

    def test_client_parse_chat(self):
        parser = build_parser()
        args = parser.parse_args(
            ["client", "chat", "--audio", "./sample.wav", "请转写"]
        )
        assert args.command == "client"
        assert args.client_action == "chat"
        assert args.audio == ["./sample.wav"]
        assert args.text == ["请转写"]

    def test_client_parse_transcribe(self):
        parser = build_parser()
        args = parser.parse_args(
            ["client", "transcribe", "./sample.wav", "--response-format", "text"]
        )
        assert args.command == "client"
        assert args.client_action == "transcribe"
        assert args.audio == ["./sample.wav"]
        assert args.response_format == "text"

    def test_client_parse_transcribe_machine_long_audio(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "client",
                "transcribe",
                "./meeting.mp3",
                "--long-audio-mode",
                "auto",
                "--per-instance-parallelism-cap",
                "3",
                "--json",
            ]
        )
        assert args.command == "client"
        assert args.client_action == "transcribe"
        assert args.long_audio_mode == "auto"
        assert args.per_instance_parallelism_cap == 3
        assert args.json is True

    def test_client_parse_transcribe_machine_long_audio_default_parallelism_cap(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "client",
                "transcribe",
                "./meeting.mp3",
                "--long-audio-mode",
                "auto",
                "--json",
            ]
        )

        assert args.command == "client"
        assert args.client_action == "transcribe"
        assert args.long_audio_mode == "auto"
        assert args.per_instance_parallelism_cap == 4
        assert args.json is True

    def test_client_parse_transcribe_long(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "client",
                "transcribe-long",
                "./meeting.mp3",
                "--target-chunk-sec",
                "75",
                "--overlap-sec",
                "5",
                "--json",
            ]
        )
        assert args.command == "client"
        assert args.client_action == "transcribe-long"
        assert args.audio == "./meeting.mp3"
        assert args.target_chunk_sec == 75.0
        assert args.overlap_sec == 5.0
        assert args.json is True

    def test_benchmark_parse(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "benchmark",
                "run",
                "-E",
                "http://localhost:27900",
                "-n",
                "10",
                "--audio",
                "./sample.wav",
            ]
        )
        assert args.command == "benchmark"
        assert args.benchmark_action == "run"
        assert args.num_samples == 10
        assert args.audio == ["./sample.wav"]
