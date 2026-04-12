"""Tests for tfmx.qsrs.benchmark."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tfmx.qsrs.benchmark import ASRBenchmarkMetrics
from tfmx.qsrs.benchmark import QSRBenchmark


class TestASRBenchmarkMetrics:
    def test_to_dict(self):
        metrics = ASRBenchmarkMetrics(n_samples=10, mode="transcribe")
        metrics.successful_requests = 8
        metrics.failed_requests = 2
        metrics.success_rate = 0.8
        payload = metrics.to_dict()
        assert payload["config"]["n_samples"] == 10
        assert payload["requests"]["successful"] == 8


class TestQSRBenchmark:
    @patch("tfmx.qsrs.benchmark.QSRClientsWithStats")
    def test_init(self, mock_clients):
        benchmark = QSRBenchmark(endpoints=["http://localhost:27900"])
        assert benchmark.max_tokens == 512
        mock_clients.assert_called_once()

    @patch("tfmx.qsrs.benchmark.QSRClientsWithStats")
    def test_check_health(self, mock_clients):
        client = MagicMock()
        client.health.return_value = MagicMock(
            healthy_machines=1,
            total_machines=1,
            healthy_instances=1,
            total_instances=1,
        )
        mock_clients.return_value = client
        benchmark = QSRBenchmark(endpoints=["http://localhost:27900"])
        assert benchmark.check_health() is True

    @patch("tfmx.qsrs.benchmark.QSRClientsWithStats")
    def test_run_collects_latency_for_chat(self, mock_clients):
        response = SimpleNamespace(
            text="hello",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            _latency_sec=0.25,
        )
        client = MagicMock()
        client.chat_batch_outcomes.return_value = [
            SimpleNamespace(
                response=response,
                error=None,
                latency_sec=0.25,
                first_token_latency_sec=0.1,
                succeeded=True,
            )
        ]
        mock_clients.return_value = client

        benchmark = QSRBenchmark(endpoints=["http://localhost:27900"], mode="chat")
        metrics = benchmark.run([{"messages": []}])

        assert metrics.request_times == [0.25]
        assert metrics.ttft_times == [0.1]
        assert metrics.total_tokens == 15
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0

    @patch("tfmx.qsrs.benchmark.QSRClientsWithStats")
    def test_run_tracks_failures_for_transcribe(self, mock_clients):
        response = SimpleNamespace(text="hello world")
        client = MagicMock()
        client.transcribe_batch_outcomes.return_value = [
            SimpleNamespace(
                response=response,
                error=None,
                latency_sec=0.2,
                first_token_latency_sec=0.0,
                succeeded=True,
            ),
            SimpleNamespace(
                response=None,
                error=ValueError("boom"),
                latency_sec=0.05,
                first_token_latency_sec=0.0,
                succeeded=False,
            ),
        ]
        mock_clients.return_value = client

        benchmark = QSRBenchmark(endpoints=["http://localhost:27900"])
        metrics = benchmark.run([{"audio": "a.wav"}, {"audio": "b.wav"}])

        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.success_rate == 0.5

    def test_build_requests(self):
        requests = QSRBenchmark.build_transcription_requests(
            ["a.wav", "b.wav"],
            model="qwen3-asr-0.6b",
        )
        assert len(requests) == 2
        assert requests[0]["model"] == "qwen3-asr-0.6b"
