"""Tests for tfmx.qwns.benchmark."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tfmx.qwns.benchmark import BenchmarkMetrics
from tfmx.qwns.benchmark import QWNBenchmark


class TestBenchmarkMetrics:
    def test_to_dict(self):
        metrics = BenchmarkMetrics(n_samples=10, max_tokens=128, temperature=0.1)
        payload = metrics.to_dict()
        assert payload["config"]["n_samples"] == 10


class TestQWNBenchmark:
    @patch("tfmx.qwns.benchmark.QWNClientsWithStats")
    def test_init(self, mock_clients):
        benchmark = QWNBenchmark(endpoints=["http://localhost:27800"])
        assert benchmark.max_tokens == 128
        mock_clients.assert_called_once()

    @patch("tfmx.qwns.benchmark.QWNClientsWithStats")
    def test_check_health(self, mock_clients):
        client = MagicMock()
        client.health.return_value = MagicMock(
            healthy_machines=1,
            total_machines=1,
            healthy_instances=1,
            total_instances=1,
        )
        mock_clients.return_value = client
        benchmark = QWNBenchmark(endpoints=["http://localhost:27800"])
        assert benchmark.check_health() is True

    @patch("tfmx.qwns.benchmark.QWNClientsWithStats")
    def test_run_collects_latency(self, mock_clients):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            _latency_sec=0.25,
        )
        client = MagicMock()
        client.chat_batch.return_value = [response]
        mock_clients.return_value = client

        benchmark = QWNBenchmark(endpoints=["http://localhost:27800"])
        metrics = benchmark.run([{"messages": []}])

        assert metrics.request_times == [0.25]
        assert metrics.total_tokens == 15

    def test_build_requests(self):
        requests = QWNBenchmark.build_requests(["hello", "world"], model="4b:4bit")
        assert len(requests) == 2
        assert requests[0]["model"] == "4b:4bit"
