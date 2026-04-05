"""Tests for tfmx.qwns.machine."""

import asyncio

from pathlib import Path
from unittest.mock import MagicMock, patch

from tfmx.qwns.machine import ModelInfo
from tfmx.qwns.machine import ModelsResponse
from tfmx.qwns.machine import QWNInstance
from tfmx.qwns.machine import QWNInstanceDiscovery
from tfmx.qwns.machine import QWNMachineDaemon
from tfmx.qwns.machine import QWNMachineServer
from tfmx.qwns.machine import QWNStatsData


class TestQWNInstance:
    def _make(self, **kwargs) -> QWNInstance:
        params = {
            "container_name": "qwn-multi--gpu0",
            "host": "localhost",
            "port": 27880,
            "gpu_id": 0,
        }
        params.update(kwargs)
        return QWNInstance(**params)

    def test_endpoint(self):
        instance = self._make()
        assert instance.endpoint == "http://localhost:27880"

    def test_available_slots(self):
        instance = self._make()
        assert instance.available_slots > 0

    def test_to_info(self):
        instance = self._make(healthy=True, model_name="4b:4bit")
        info = instance.to_info()
        assert info.healthy is True
        assert info.model_name == "4b:4bit"


class TestQWNStatsData:
    def test_to_model(self):
        stats = QWNStatsData(
            total_requests=3, total_tokens=50, total_errors=1, active_requests=2
        )
        model = stats.to_model()
        assert model.total_requests == 3
        assert model.active_requests == 2


class TestQWNInstanceDiscovery:
    @patch("subprocess.run")
    def test_discover(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="qwn-multi--gpu0|vllm/vllm-openai:latest|0.0.0.0:27880->8000/tcp\n",
        )
        instances = QWNInstanceDiscovery.discover()
        assert len(instances) == 1
        assert instances[0].gpu_id == 0

    def test_from_endpoints(self):
        instances = QWNInstanceDiscovery.from_endpoints(
            ["http://localhost:27880", "27881"]
        )
        assert len(instances) == 2
        assert instances[0].port == 27880
        assert instances[1].port == 27881


class TestQWNMachineServer:
    def test_get_model_label(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            gpu_id=0,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[instance])
        assert server._get_model_label(instance) == "4b:4bit"

    def test_rewrite_model_in_body(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[instance])
        body = b'{"model":"4b","messages":[{"role":"user","content":"hello"}]}'
        rewritten = server._rewrite_model_in_body(body, instance)
        assert b"4b:4bit" in rewritten

    def test_models_endpoint(self):
        instance = QWNInstance(
            container_name="qwn-multi--gpu0",
            host="localhost",
            port=27880,
            healthy=True,
            model_name="4b:4bit",
        )
        server = QWNMachineServer(instances=[instance])
        result = asyncio.run(server.models())
        assert isinstance(result, ModelsResponse)
        assert result.data[0].id == "4b:4bit"


class TestQWNMachineDaemon:
    def test_get_pid_missing(self, tmp_path):
        daemon = QWNMachineDaemon(
            pid_file=tmp_path / "qwn.pid",
            log_file=tmp_path / "qwn.log",
        )
        assert daemon.get_pid() is None

    @patch("subprocess.Popen")
    def test_start_background_strips_cli_machine_arg(self, mock_popen, tmp_path):
        proc = MagicMock(pid=4321)
        mock_popen.return_value = proc
        daemon = QWNMachineDaemon(
            pid_file=tmp_path / "qwn.pid",
            log_file=tmp_path / "qwn.log",
        )

        daemon.start_background(["cli.py", "machine", "run", "-b"])

        cmd = mock_popen.call_args.args[0]
        assert cmd[1:] == ["-m", "tfmx.qwns.machine", "run"]
        assert daemon.pid_file.read_text().strip() == "4321"
