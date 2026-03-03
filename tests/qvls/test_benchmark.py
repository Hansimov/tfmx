"""Tests for tfmx.qvls.benchmark module"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from tfmx.qvls.benchmark import (
    BenchmarkMetrics,
    QVLBenchmark,
)


# ── BenchmarkMetrics ─────────────────────────────────────────────────


class TestBenchmarkMetrics:
    """Test BenchmarkMetrics dataclass."""

    def test_defaults(self):
        m = BenchmarkMetrics()
        assert m.n_samples == 0
        assert m.max_tokens == 512
        assert m.temperature == 0.1
        assert m.text_only is False
        assert m.total_time == 0.0
        assert m.request_times == []
        assert m.requests_per_second == 0.0
        assert m.total_tokens == 0

    def test_calculate_latency_percentiles(self):
        m = BenchmarkMetrics()
        m.request_times = [0.1, 0.2, 0.3, 0.4, 0.5]
        m.calculate_latency_percentiles()
        assert m.latency_min == 0.1
        assert m.latency_max == 0.5
        assert abs(m.latency_avg - 0.3) < 1e-9
        assert m.latency_p50 == 0.3  # index 2

    def test_calculate_latency_percentiles_empty(self):
        m = BenchmarkMetrics()
        m.calculate_latency_percentiles()
        assert m.latency_min == 0.0
        assert m.latency_max == 0.0

    def test_calculate_latency_p95_small_sample(self):
        """p95 falls back to max for small samples (n <= 20)."""
        m = BenchmarkMetrics()
        m.request_times = [float(i) for i in range(1, 11)]
        m.calculate_latency_percentiles()
        assert m.latency_p95 == 10.0  # max
        assert m.latency_p99 == 10.0  # max

    def test_calculate_latency_p95_large_sample(self):
        """p95 computed properly for large samples."""
        m = BenchmarkMetrics()
        m.request_times = [float(i) / 100 for i in range(100)]
        m.calculate_latency_percentiles()
        assert m.latency_p50 == pytest.approx(0.50, abs=0.01)
        assert m.latency_p95 == pytest.approx(0.95, abs=0.01)

    def test_to_dict_structure(self):
        m = BenchmarkMetrics(
            n_samples=50,
            max_tokens=128,
            temperature=0.1,
            text_only=True,
            endpoints=["http://localhost:29800"],
            total_time=10.5,
            requests_per_second=4.76,
            gen_tokens_per_second=610.0,
            total_tokens_per_second=1200.0,
            total_prompt_tokens=6000,
            total_gen_tokens=6400,
            total_tokens=12400,
        )
        d = m.to_dict()

        assert "config" in d
        assert d["config"]["n_samples"] == 50
        assert d["config"]["text_only"] is True
        assert d["config"]["endpoints"] == ["http://localhost:29800"]

        assert "timing" in d
        assert d["timing"]["total_time_sec"] == 10.5

        assert "throughput" in d
        assert d["throughput"]["requests_per_second"] == 4.76

        assert "tokens" in d
        assert d["tokens"]["total_gen_tokens"] == 6400

        assert "latency_sec" in d

    def test_to_dict_rounds_values(self):
        m = BenchmarkMetrics(
            total_time=1.23456789,
            requests_per_second=3.14159,
            latency_avg=0.12345,
        )
        d = m.to_dict()
        assert d["timing"]["total_time_sec"] == 1.235
        assert d["throughput"]["requests_per_second"] == 3.14
        assert d["latency_sec"]["avg"] == 0.1235


# ── QVLBenchmark ─────────────────────────────────────────────────────


class TestQVLBenchmark:
    """Test QVLBenchmark runner (mocked client layer)."""

    @patch("tfmx.qvls.benchmark.QVLClientsWithStats")
    def test_init(self, mock_clients_cls):
        b = QVLBenchmark(endpoints=["http://localhost:29800"])
        assert b.endpoints == ["http://localhost:29800"]
        assert b.max_tokens == 128
        assert b.temperature == 0.1
        mock_clients_cls.assert_called_once()

    @patch("tfmx.qvls.benchmark.QVLClientsWithStats")
    def test_context_manager(self, mock_clients_cls):
        mock_client = MagicMock()
        mock_clients_cls.return_value = mock_client

        with QVLBenchmark(endpoints=["http://localhost:29800"]) as b:
            assert b.clients is mock_client
        mock_client.close.assert_called_once()

    @patch("tfmx.qvls.benchmark.QVLClientsWithStats")
    def test_check_health_healthy(self, mock_clients_cls):
        mock_client = MagicMock()
        mock_client.health.return_value = MagicMock(
            healthy_machines=2,
            total_machines=2,
            healthy_instances=6,
            total_instances=6,
            status="healthy",
        )
        mock_clients_cls.return_value = mock_client

        b = QVLBenchmark(endpoints=["http://a:29800", "http://b:29800"])
        assert b.check_health() is True

    @patch("tfmx.qvls.benchmark.QVLClientsWithStats")
    def test_check_health_unhealthy(self, mock_clients_cls):
        mock_client = MagicMock()
        mock_client.health.return_value = MagicMock(
            healthy_machines=0,
            total_machines=1,
            healthy_instances=0,
            total_instances=3,
            status="unhealthy",
        )
        mock_clients_cls.return_value = mock_client

        b = QVLBenchmark(endpoints=["http://localhost:29800"])
        assert b.check_health() is False

    @patch("tfmx.qvls.benchmark.QVLClientsWithStats")
    def test_run_collects_metrics(self, mock_clients_cls):
        """Benchmark run collects token stats from responses."""
        mock_usage = MagicMock(
            prompt_tokens=50, completion_tokens=100, total_tokens=150
        )
        mock_resp = MagicMock(usage=mock_usage)
        mock_client = MagicMock()
        mock_client.chat_batch.return_value = [mock_resp] * 10
        mock_clients_cls.return_value = mock_client

        b = QVLBenchmark(endpoints=["http://localhost:29800"], max_tokens=128)
        requests = [
            {"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(10)
        ]
        metrics = b.run(requests, text_only=True)

        assert metrics.n_samples == 10
        assert metrics.total_prompt_tokens == 500  # 50 * 10
        assert metrics.total_gen_tokens == 1000  # 100 * 10
        assert metrics.total_tokens == 1500  # 150 * 10
        assert metrics.total_time > 0
        assert metrics.requests_per_second > 0

    @patch("tfmx.qvls.benchmark.QVLClientsWithStats")
    def test_run_handles_exception(self, mock_clients_cls):
        """Benchmark run handles client exceptions gracefully."""
        mock_client = MagicMock()
        mock_client.chat_batch.side_effect = RuntimeError("connection refused")
        mock_clients_cls.return_value = mock_client

        b = QVLBenchmark(endpoints=["http://localhost:29800"])
        requests = [{"messages": [{"role": "user", "content": "test"}]}]
        metrics = b.run(requests)

        assert metrics.total_tokens == 0
        assert metrics.total_time > 0

    @patch("tfmx.qvls.benchmark.QVLClientsWithStats")
    def test_run_empty_requests(self, mock_clients_cls):
        mock_client = MagicMock()
        mock_client.chat_batch.return_value = []
        mock_clients_cls.return_value = mock_client

        b = QVLBenchmark(endpoints=["http://localhost:29800"])
        metrics = b.run([])
        assert metrics.n_samples == 0
