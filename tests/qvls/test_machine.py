"""Tests for tfmx.qvls.machine module"""

import asyncio
import os
import signal
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import asdict

from tfmx.qvls.machine import (
    VLLMInstance,
    VLLMStatsData,
    VLLMInstanceDiscovery,
    QVLMachineServer,
    QVLMachineDaemon,
    ChatCompletionRequest,
    HealthResponse,
    InstanceInfo,
    MachineStats,
    InfoResponse,
    VLLM_CONTAINER_IMAGE_PATTERN,
)
from tfmx.qvls.compose import MAX_CONCURRENT_REQUESTS, MACHINE_PORT, SERVER_PORT


# ── VLLMInstance ──────────────────────────────────────────────────────


class TestVLLMInstance:
    """Test VLLMInstance dataclass and properties."""

    def _make(self, **kw) -> VLLMInstance:
        defaults = dict(
            container_name="qvl-multi--gpu0",
            host="localhost",
            port=29880,
            gpu_id=0,
        )
        defaults.update(kw)
        return VLLMInstance(**defaults)

    def test_endpoint(self):
        inst = self._make()
        assert inst.endpoint == "http://localhost:29880"

    def test_chat_url(self):
        inst = self._make()
        assert inst.chat_url == "http://localhost:29880/v1/chat/completions"

    def test_health_url(self):
        inst = self._make()
        assert inst.health_url == "http://localhost:29880/health"

    def test_models_url(self):
        inst = self._make()
        assert inst.models_url == "http://localhost:29880/v1/models"

    def test_is_idle_no_requests(self):
        inst = self._make()
        assert inst.is_idle is True

    def test_is_idle_at_max(self):
        inst = self._make()
        inst._active_requests = MAX_CONCURRENT_REQUESTS
        assert inst.is_idle is False

    def test_available_slots(self):
        inst = self._make()
        assert inst.available_slots == MAX_CONCURRENT_REQUESTS

    def test_available_slots_some_active(self):
        inst = self._make()
        inst._active_requests = 2
        assert inst.available_slots == MAX_CONCURRENT_REQUESTS - 2

    def test_available_slots_negative_clamped(self):
        inst = self._make()
        inst._active_requests = MAX_CONCURRENT_REQUESTS + 10
        assert inst.available_slots == 0

    def test_repr_healthy(self):
        inst = self._make(healthy=True)
        r = repr(inst)
        assert "✓" in r
        assert "qvl-multi--gpu0" in r
        assert "GPU0" in r

    def test_repr_unhealthy(self):
        inst = self._make(healthy=False)
        r = repr(inst)
        assert "×" in r

    def test_repr_with_model(self):
        inst = self._make(
            healthy=True,
            model_name="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
            quant_level="4bit",
        )
        r = repr(inst)
        assert "4bit" in r.lower()

    def test_repr_no_gpu(self):
        inst = self._make(gpu_id=None)
        r = repr(inst)
        assert "GPU?" in r

    def test_to_info(self):
        inst = self._make(
            healthy=True,
            model_name="cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit",
            quant_method="awq",
            quant_level="4bit",
        )
        info = inst.to_info()
        assert isinstance(info, InstanceInfo)
        assert info.healthy is True
        assert info.gpu_id == 0
        assert "4bit" in info.model_label
        assert info.quant_method == "awq"

    def test_to_info_no_model(self):
        inst = self._make(healthy=False)
        info = inst.to_info()
        assert info.model_label == ""
        assert info.healthy is False


# ── VLLMStatsData ────────────────────────────────────────────────────


class TestVLLMStatsData:
    """Test VLLMStatsData internal statistics tracking."""

    def test_defaults(self):
        stats = VLLMStatsData()
        assert stats.total_requests == 0
        assert stats.total_tokens == 0
        assert stats.total_errors == 0
        assert stats.active_requests == 0
        assert stats.requests_per_instance == {}

    def test_to_model(self):
        stats = VLLMStatsData(
            total_requests=100,
            total_tokens=5000,
            total_errors=3,
            active_requests=2,
            requests_per_instance={"gpu0": 50, "gpu1": 50},
        )
        model = stats.to_model()
        assert isinstance(model, MachineStats)
        assert model.total_requests == 100
        assert model.total_tokens == 5000
        assert model.total_errors == 3
        assert model.active_requests == 2
        assert model.requests_per_instance == {"gpu0": 50, "gpu1": 50}


# ── VLLMInstanceDiscovery ────────────────────────────────────────────


class TestVLLMInstanceDiscovery:
    """Test Docker container discovery."""

    def test_extract_host_port_bridge_mode(self):
        port = VLLMInstanceDiscovery._extract_host_port(
            "0.0.0.0:29880->8000/tcp", "test"
        )
        assert port == 29880

    def test_extract_host_port_ipv6(self):
        port = VLLMInstanceDiscovery._extract_host_port(":::29882->8000/tcp", "test")
        assert port == 29882

    def test_extract_host_port_no_match(self):
        port = VLLMInstanceDiscovery._extract_host_port("", "")
        assert port is None

    def test_extract_gpu_id(self):
        assert VLLMInstanceDiscovery._extract_gpu_id("qvl-multi--gpu3") == 3

    def test_extract_gpu_id_no_match(self):
        assert VLLMInstanceDiscovery._extract_gpu_id("random-container") is None

    def test_extract_gpu_id_multi_digit(self):
        assert VLLMInstanceDiscovery._extract_gpu_id("qvl--gpu12") == 12

    def test_from_endpoints_url_format(self):
        endpoints = [
            "http://localhost:29880",
            "http://localhost:29881",
        ]
        instances = VLLMInstanceDiscovery.from_endpoints(endpoints)
        assert len(instances) == 2
        assert instances[0].host == "localhost"
        assert instances[0].port == 29880
        assert instances[1].port == 29881

    def test_from_endpoints_port_only(self):
        instances = VLLMInstanceDiscovery.from_endpoints(["29880", "29881"])
        assert len(instances) == 2
        assert instances[0].host == "localhost"
        assert instances[0].port == 29880
        assert instances[0].container_name == "manual-0"

    def test_from_endpoints_empty(self):
        instances = VLLMInstanceDiscovery.from_endpoints(["", "  "])
        assert len(instances) == 0

    def test_from_endpoints_invalid(self):
        instances = VLLMInstanceDiscovery.from_endpoints(["not-a-url"])
        assert len(instances) == 0

    def test_from_endpoints_mixed(self):
        """Mix valid and invalid endpoint specs."""
        instances = VLLMInstanceDiscovery.from_endpoints(
            ["http://localhost:29880", "invalid", "29882"]
        )
        assert len(instances) == 2

    @patch("subprocess.run")
    def test_discover_basic(self, mock_run):
        """Discover vLLM containers from docker ps."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "qvl-multi--gpu0|vllm/vllm-openai:latest|0.0.0.0:29880->8000/tcp\n"
                "qvl-multi--gpu1|vllm/vllm-openai:latest|0.0.0.0:29881->8000/tcp\n"
            ),
        )
        instances = VLLMInstanceDiscovery.discover()
        assert len(instances) == 2
        assert instances[0].gpu_id == 0
        assert instances[0].port == 29880
        assert instances[1].gpu_id == 1

    @patch("subprocess.run")
    def test_discover_filters_non_vllm(self, mock_run):
        """Non-vLLM containers are filtered out."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "qvl-multi--gpu0|vllm/vllm-openai:latest|0.0.0.0:29880->8000/tcp\n"
                "tei-multi--gpu0|ghcr.io/huggingface/tei:latest|0.0.0.0:28880->80/tcp\n"
            ),
        )
        instances = VLLMInstanceDiscovery.discover()
        assert len(instances) == 1
        assert instances[0].container_name == "qvl-multi--gpu0"

    @patch("subprocess.run")
    def test_discover_sorted_by_gpu(self, mock_run):
        """Instances sorted by GPU ID."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "qvl-multi--gpu2|vllm/vllm-openai:latest|0.0.0.0:29882->8000/tcp\n"
                "qvl-multi--gpu0|vllm/vllm-openai:latest|0.0.0.0:29880->8000/tcp\n"
            ),
        )
        instances = VLLMInstanceDiscovery.discover()
        assert instances[0].gpu_id == 0
        assert instances[1].gpu_id == 2

    @patch("subprocess.run")
    def test_discover_docker_failure(self, mock_run):
        """Returns empty on docker failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        instances = VLLMInstanceDiscovery.discover()
        assert instances == []

    @patch("subprocess.run")
    def test_discover_no_containers(self, mock_run):
        """Returns empty when no containers running."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        instances = VLLMInstanceDiscovery.discover()
        assert instances == []

    @patch("subprocess.run")
    def test_discover_name_pattern(self, mock_run):
        """Name pattern filter."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "qvl-multi--gpu0|vllm/vllm-openai:latest|0.0.0.0:29880->8000/tcp\n"
                "qvl-other--gpu1|vllm/vllm-openai:latest|0.0.0.0:29881->8000/tcp\n"
            ),
        )
        instances = VLLMInstanceDiscovery.discover(name_pattern="qvl-multi")
        # The name_pattern also filters in code via regex
        assert all("qvl-multi" in i.container_name for i in instances)


# ── Pydantic Models ──────────────────────────────────────────────────


class TestPydanticModels:
    """Test Pydantic request/response models."""

    def test_chat_completion_request_defaults(self):
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        assert req.model == ""
        assert req.max_tokens == 512
        assert req.temperature == 0.7
        assert req.stream is False

    def test_chat_completion_request_custom(self):
        req = ChatCompletionRequest(
            model="8b-instruct:4bit",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=256,
            temperature=0.1,
        )
        assert req.model == "8b-instruct:4bit"
        assert req.max_tokens == 256

    def test_health_response(self):
        hr = HealthResponse(status="healthy", healthy=6, total=6)
        assert hr.status == "healthy"

    def test_instance_info(self):
        ii = InstanceInfo(
            name="qvl--gpu0",
            endpoint="http://localhost:29880",
            gpu_id=0,
            healthy=True,
            model_label="8B-Instruct:4bit",
        )
        assert ii.name == "qvl--gpu0"
        assert ii.model_label == "8B-Instruct:4bit"

    def test_machine_stats_defaults(self):
        ms = MachineStats()
        assert ms.total_requests == 0
        assert ms.requests_per_instance == {}

    def test_info_response(self):
        ir = InfoResponse(
            port=29800,
            instances=[],
            stats=MachineStats(),
            available_models=["8B-Instruct:4bit"],
        )
        assert ir.port == 29800
        assert ir.available_models == ["8B-Instruct:4bit"]


# ── QVLMachineServer ─────────────────────────────────────────────────


class TestQVLMachineServer:
    """Test QVLMachineServer logic (no actual HTTP)."""

    def _make_instances(self, n=3) -> list[VLLMInstance]:
        # Use canonical base model names (as discovered by /v1/models).
        # In production, vLLM reports the model_id which is the base name,
        # not the AWQ repo name.
        base_names = [
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
        ]
        return [
            VLLMInstance(
                container_name=f"qvl-multi--gpu{i}",
                host="localhost",
                port=29880 + i,
                gpu_id=i,
                healthy=True,
                model_name=base_names[i % 3],
                quant_method="awq",
                quant_level="4bit",
            )
            for i in range(n)
        ]

    def test_init(self):
        instances = self._make_instances()
        server = QVLMachineServer(instances=instances, port=29800)
        assert server.port == 29800
        assert len(server.instances) == 3
        assert server.timeout == 120.0

    def test_get_healthy_instances(self):
        instances = self._make_instances()
        instances[1].healthy = False
        server = QVLMachineServer(instances=instances)
        healthy = server.get_healthy_instances()
        assert len(healthy) == 2

    def test_get_idle_instance_all_idle(self):
        instances = self._make_instances()
        server = QVLMachineServer(instances=instances)
        server._build_router()
        inst = server._get_idle_instance()
        assert inst is not None
        assert inst.healthy is True

    def test_get_idle_instance_none_idle(self):
        instances = self._make_instances()
        for inst in instances:
            inst._active_requests = MAX_CONCURRENT_REQUESTS
        server = QVLMachineServer(instances=instances)
        server._build_router()
        result = server._get_idle_instance()
        assert result is None

    def test_get_idle_instance_unhealthy_skipped(self):
        instances = self._make_instances(1)
        instances[0].healthy = False
        server = QVLMachineServer(instances=instances)
        server._build_router()
        result = server._get_idle_instance()
        assert result is None

    def test_get_idle_instance_most_available_slots(self):
        """Should return instance with most available slots."""
        instances = self._make_instances(3)
        instances[0]._active_requests = 3
        instances[1]._active_requests = 1
        instances[2]._active_requests = 2
        server = QVLMachineServer(instances=instances)
        server._build_router()
        inst = server._get_idle_instance()
        assert inst.container_name == instances[1].container_name

    def test_build_router(self):
        instances = self._make_instances()
        server = QVLMachineServer(instances=instances)
        server._build_router()
        assert len(server.router.instances) == 3
        models = server.router.get_available_models()
        assert len(models) >= 1

    def test_get_idle_instance_with_model_filter(self):
        """Filter by model shortcut finds matching instances."""
        instances = self._make_instances(3)
        server = QVLMachineServer(instances=instances)
        server._build_router()
        inst = server._get_idle_instance(model="2b-instruct")
        assert inst is not None
        assert "2B" in inst.model_name

    def test_get_idle_instance_model_no_match(self):
        """Non-matching model returns None."""
        instances = self._make_instances(1)
        # instances[0].model_name is already "Qwen/Qwen3-VL-2B-Instruct"
        server = QVLMachineServer(instances=instances)
        server._build_router()
        inst = server._get_idle_instance(model="8b-thinking")
        assert inst is None

    def test_perf_tracking_flag(self):
        instances = self._make_instances(1)
        server = QVLMachineServer(instances=instances, enable_perf_tracking=True)
        assert server.enable_perf_tracking is True

    def test_app_has_routes(self):
        instances = self._make_instances(1)
        server = QVLMachineServer(instances=instances)
        route_paths = [r.path for r in server.app.routes]
        assert "/health" in route_paths
        assert "/info" in route_paths
        assert "/v1/chat/completions" in route_paths
        assert "/chat" in route_paths


# ── QVLMachineDaemon ─────────────────────────────────────────────────


class TestQVLMachineDaemon:
    """Test daemon lifecycle management (PID/log file operations)."""

    @pytest.fixture(autouse=True)
    def daemon(self, tmp_path):
        """Create a daemon with temp PID/log files for each test."""
        self.pid_file = tmp_path / "qvl_machine.pid"
        self.log_file = tmp_path / "qvl_machine.log"
        self.d = QVLMachineDaemon(
            pid_file=self.pid_file,
            log_file=self.log_file,
        )

    # ── get_pid ──

    def test_get_pid_no_file(self):
        assert self.d.get_pid() is None

    def test_get_pid_valid(self):
        """Reading PID of current process should succeed."""
        self.pid_file.write_text(str(os.getpid()))
        assert self.d.get_pid() == os.getpid()

    def test_get_pid_stale_cleans_up(self):
        """Stale PID (nonexistent process) returns None and removes file."""
        self.pid_file.write_text("999999999")
        assert self.d.get_pid() is None
        assert not self.pid_file.exists()

    def test_get_pid_garbage_content(self):
        """Non-integer PID file content returns None and removes file."""
        self.pid_file.write_text("not-a-pid")
        assert self.d.get_pid() is None
        assert not self.pid_file.exists()

    # ── is_running ──

    def test_is_running_false_no_file(self):
        assert self.d.is_running() is False

    def test_is_running_true(self):
        self.pid_file.write_text(str(os.getpid()))
        assert self.d.is_running() is True

    def test_is_running_false_stale(self):
        self.pid_file.write_text("999999999")
        assert self.d.is_running() is False

    # ── write_pid / remove_pid ──

    def test_write_pid(self):
        self.d.write_pid()
        assert self.pid_file.exists()
        assert int(self.pid_file.read_text().strip()) == os.getpid()

    def test_remove_pid(self):
        self.d.write_pid()
        self.d.remove_pid()
        assert not self.pid_file.exists()

    def test_remove_pid_no_file(self):
        """remove_pid is safe to call when no PID file exists."""
        self.d.remove_pid()  # should not raise

    # ── stop ──

    def test_stop_no_daemon(self):
        """Stop returns False when no daemon is running."""
        assert self.d.stop() is False

    @patch("os.kill")
    def test_stop_sends_sigterm(self, mock_kill):
        """Stop sends SIGTERM to the daemon PID."""
        self.pid_file.write_text("12345")
        # First os.kill(pid, 0) in get_pid — process alive
        # Then os.kill(pid, SIGTERM) — send signal
        # Then os.kill(pid, 0) in the wait loop — process gone
        call_count = 0

        def side_effect(pid, sig):
            nonlocal call_count
            call_count += 1
            if sig == 0 and call_count == 1:
                return  # alive for get_pid check
            if sig == signal.SIGTERM:
                return  # SIGTERM sent ok
            if sig == 0:
                raise ProcessLookupError  # dead after SIGTERM

        mock_kill.side_effect = side_effect
        result = self.d.stop()
        assert result is True
        assert not self.pid_file.exists()

    # ── status (output verification) ──

    def test_status_not_running(self, capsys):
        """status prints 'not running' when no daemon active."""
        self.d.status()
        # No exception raised

    def test_status_running(self):
        """status works when daemon PID is alive."""
        self.pid_file.write_text(str(os.getpid()))
        self.d.status()  # should not raise

    def test_status_with_log_file(self):
        """status reports log size when log file exists."""
        self.pid_file.write_text(str(os.getpid()))
        self.log_file.write_text("sample log line\n" * 100)
        self.d.status()  # should not raise

    # ── show_logs ──

    def test_show_logs_no_file(self):
        """show_logs handles missing log file gracefully."""
        self.d.show_logs()  # should not raise

    @patch("subprocess.run")
    def test_show_logs_reads_tail(self, mock_run):
        """show_logs calls tail with correct args."""
        self.log_file.write_text("line1\nline2\nline3\n")
        mock_run.return_value = MagicMock(stdout="line1\nline2\nline3\n")
        self.d.show_logs(follow=False, tail=20)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "tail" in cmd
        assert "-n" in cmd
        assert "20" in cmd

    @patch("subprocess.run")
    def test_show_logs_follow(self, mock_run):
        """show_logs uses tail -f in follow mode."""
        self.log_file.write_text("log\n")
        self.d.show_logs(follow=True, tail=10)
        cmd = mock_run.call_args[0][0]
        assert "-f" in cmd

    # ── start_background ──

    @patch("subprocess.Popen")
    def test_start_background_strips_b_flag(self, mock_popen):
        """start_background removes -b from argv."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_popen.return_value = mock_proc

        self.d.start_background(["qvl_machine", "run", "-b", "-p", "29800"])

        cmd = mock_popen.call_args[0][0]
        assert "-b" not in cmd
        assert "--background" not in cmd
        assert "-p" in cmd
        assert "29800" in cmd

    @patch("subprocess.Popen")
    def test_start_background_writes_pid(self, mock_popen):
        """start_background writes child PID to file."""
        mock_proc = MagicMock()
        mock_proc.pid = 42000
        mock_popen.return_value = mock_proc

        self.d.start_background(["qvl_machine", "run", "-b"])
        assert self.pid_file.exists()
        assert int(self.pid_file.read_text().strip()) == 42000

    @patch("subprocess.Popen")
    def test_start_background_sets_env(self, mock_popen):
        """start_background passes _QVL_MACHINE_DAEMON env var."""
        mock_proc = MagicMock()
        mock_proc.pid = 11111
        mock_popen.return_value = mock_proc

        self.d.start_background(["qvl_machine", "run", "-b"])
        env = mock_popen.call_args[1]["env"]
        assert env["_QVL_MACHINE_DAEMON"] == "1"

    @patch("subprocess.Popen")
    def test_start_background_detached(self, mock_popen):
        """start_background uses start_new_session=True for detach."""
        mock_proc = MagicMock()
        mock_proc.pid = 22222
        mock_popen.return_value = mock_proc

        self.d.start_background(["qvl_machine", "run", "--background"])
        assert mock_popen.call_args[1]["start_new_session"] is True
