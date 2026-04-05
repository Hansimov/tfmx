"""Tests for verifying the refactored module structure and imports"""

import pytest


class TestPackageImports:
    """Verify all public APIs are importable from expected locations"""

    def test_import_root_package(self):
        import tfmx

        assert tfmx is not None

    def test_import_utils_module(self):
        from tfmx.utils import floats_to_bits, bits_to_hash, bits_dist, hash_dist
        from tfmx.utils import bits_sim, hash_sim, dot_sim
        from tfmx.utils import LSHConverter

        assert LSHConverter is not None

    def test_import_gpus_module(self):
        from tfmx.gpus import NvidiaSettingsParser, GPUFanController, GPUFanArgParser
        from tfmx.gpus import NvidiaSmiParser, GPUPowerController, GPUPowerArgParser

        assert GPUFanController is not None
        assert GPUPowerController is not None

    def test_import_teis_module(self):
        from tfmx.teis import TEIComposer, TEIComposeArgParser
        from tfmx.teis import GPUInfo, GPUDetector
        from tfmx.teis import TEIClient, AsyncTEIClient
        from tfmx.teis import TEIClients, TEIClientsWithStats
        from tfmx.teis import MachineState, MachineScheduler
        from tfmx.teis import IdleFillingScheduler, WorkerState
        from tfmx.teis import PerfTracker, WorkerEvent, TaskRecord

        assert TEIComposer is not None
        assert TEIClient is not None

    def test_import_qvls_module(self):
        from tfmx.qvls import QVLComposer, QVLComposeArgParser
        from tfmx.qvls import QVLClient, AsyncQVLClient
        from tfmx.qvls import QVLRouter, InstanceDescriptor
        from tfmx.qvls import QVLMachineServer, VLLMInstance
        from tfmx.qvls import QVLBenchmark, BenchmarkMetrics

        assert QVLComposer is not None
        assert QVLClient is not None

    def test_import_qwns_module(self):
        from tfmx.qwns import QWNComposer
        from tfmx.qwns import QWNClient, AsyncQWNClient
        from tfmx.qwns import QWNRouter, InstanceDescriptor
        from tfmx.qwns import QWNMachineServer, QWNInstance
        from tfmx.qwns import QWNBenchmark, BenchmarkMetrics

        assert QWNComposer is not None
        assert QWNClient is not None

    def test_import_from_root(self):
        """All public APIs should also be importable from root tfmx"""
        from tfmx import TEIClient, TEIClients, TEIClientsWithStats
        from tfmx import TEIComposer
        from tfmx import floats_to_bits, bits_to_hash
        from tfmx import GPUFanController, GPUPowerController
        from tfmx import PerfTracker, IdleFillingScheduler
        from tfmx import LSHConverter
        from tfmx import QWNClient, QWNComposer, QWNBenchmark

        assert TEIClient is not None
        assert QWNClient is not None

    def test_gpus_submodules(self):
        """GPU submodule files should be importable"""
        from tfmx.gpus import ctl, fan, mon, pow

        assert ctl is not None
        assert fan is not None

    def test_teis_submodules(self):
        """TEI submodule files should be importable"""
        from tfmx.teis import client, clients, compose, scheduler
        from tfmx.teis import perf_tracker, performance
        from tfmx.teis import benchmark, benchtext

        assert client is not None

    def test_qvls_submodules(self):
        """QVL submodule files should be importable"""
        from tfmx.qvls import client, clients, compose, scheduler
        from tfmx.qvls import perf_tracker, performance
        from tfmx.qvls import benchmark, benchimgs
        from tfmx.qvls import router, machine

        assert client is not None
        assert router is not None

    def test_qwns_submodules(self):
        """QWN submodule files should be importable"""
        from tfmx.qwns import benchmark, client, clients, compose, scheduler
        from tfmx.qwns import perf_tracker, performance
        from tfmx.qwns import cli, machine, router

        assert client is not None
        assert machine is not None

    def test_utils_submodules(self):
        """Utils submodule files should be importable"""
        from tfmx.utils import lsh, vectors

        assert lsh is not None
        assert vectors is not None


class TestDeprecatedImportsRemoved:
    """Verify deprecated modules are no longer importable"""

    def test_no_llm_module(self):
        with pytest.raises(ImportError):
            from tfmx import llm

    def test_no_embed_client_module(self):
        with pytest.raises(ImportError):
            from tfmx import embed_client

    def test_no_embed_server_module(self):
        with pytest.raises(ImportError):
            from tfmx import embed_server


class TestModuleStructure:
    """Verify the physical file structure"""

    def test_gpus_directory_exists(self):
        import tfmx.gpus
        from pathlib import Path

        gpus_dir = Path(tfmx.gpus.__file__).parent
        assert (gpus_dir / "ctl.py").exists()
        assert (gpus_dir / "fan.py").exists()
        assert (gpus_dir / "mon.py").exists()
        assert (gpus_dir / "pow.py").exists()

    def test_teis_directory_exists(self):
        import tfmx.teis
        from pathlib import Path

        teis_dir = Path(tfmx.teis.__file__).parent
        assert (teis_dir / "client.py").exists()
        assert (teis_dir / "clients.py").exists()
        assert (teis_dir / "compose.py").exists()
        assert (teis_dir / "scheduler.py").exists()
        assert (teis_dir / "machine.py").exists()
        assert (teis_dir / "perf_tracker.py").exists()
        assert (teis_dir / "performance.py").exists()
        assert (teis_dir / "benchmark.py").exists()
        assert (teis_dir / "benchtext.py").exists()
        assert (teis_dir / "clients_core.py").exists()
        assert (teis_dir / "clients_cli.py").exists()
        assert (teis_dir / "clients_stats.py").exists()

    def test_qvls_directory_exists(self):
        import tfmx.qvls
        from pathlib import Path

        qvls_dir = Path(tfmx.qvls.__file__).parent
        assert (qvls_dir / "client.py").exists()
        assert (qvls_dir / "clients.py").exists()
        assert (qvls_dir / "compose.py").exists()
        assert (qvls_dir / "scheduler.py").exists()
        assert (qvls_dir / "machine.py").exists()
        assert (qvls_dir / "perf_tracker.py").exists()
        assert (qvls_dir / "performance.py").exists()
        assert (qvls_dir / "benchmark.py").exists()
        assert (qvls_dir / "benchimgs.py").exists()
        assert (qvls_dir / "clients_core.py").exists()
        assert (qvls_dir / "clients_cli.py").exists()
        assert (qvls_dir / "clients_stats.py").exists()
        assert (qvls_dir / "router.py").exists()

    def test_qwns_directory_exists(self):
        import tfmx.qwns
        from pathlib import Path

        qwns_dir = Path(tfmx.qwns.__file__).parent
        assert (qwns_dir / "client.py").exists()
        assert (qwns_dir / "clients.py").exists()
        assert (qwns_dir / "compose.py").exists()
        assert (qwns_dir / "scheduler.py").exists()
        assert (qwns_dir / "machine.py").exists()
        assert (qwns_dir / "perf_tracker.py").exists()
        assert (qwns_dir / "performance.py").exists()
        assert (qwns_dir / "benchmark.py").exists()
        assert (qwns_dir / "router.py").exists()
        assert (qwns_dir / "cli.py").exists()

    def test_utils_directory_exists(self):
        import tfmx.utils
        from pathlib import Path

        utils_dir = Path(tfmx.utils.__file__).parent
        assert (utils_dir / "lsh.py").exists()
        assert (utils_dir / "vectors.py").exists()

    def test_configs_directory_exists(self):
        import tfmx
        from pathlib import Path

        configs_dir = Path(tfmx.__file__).parent / "configs"
        assert configs_dir.is_dir()

    def test_weights_directory_exists(self):
        import tfmx
        from pathlib import Path

        weights_dir = Path(tfmx.__file__).parent / "weights"
        assert weights_dir.is_dir()

    def test_model_configs_in_teis(self):
        import tfmx.teis
        from pathlib import Path

        teis_dir = Path(tfmx.teis.__file__).parent
        assert (teis_dir / "config_qwen3_embedding_06b.json").exists()
        assert (teis_dir / "config_sentence_transformers.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
