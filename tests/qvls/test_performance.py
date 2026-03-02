"""Tests for tfmx.qvls.performance module"""

import pytest
import json
from pathlib import Path

from tfmx.qvls.performance import ExplorationConfig


class TestExplorationConfig:
    """Test ExplorationConfig persistence."""

    def test_init_no_file(self, tmp_path):
        config = ExplorationConfig(config_dir=tmp_path)
        assert config._config == {}

    def test_save_and_load(self, tmp_path):
        config1 = ExplorationConfig(config_dir=tmp_path)
        config1._config = {"test_key": {"max_concurrent": 8}}
        config1._save()

        config2 = ExplorationConfig(config_dir=tmp_path)
        assert "test_key" in config2._config
        assert config2._config["test_key"]["max_concurrent"] == 8

    def test_config_file_location(self, tmp_path):
        config = ExplorationConfig(config_dir=tmp_path)
        assert config.config_path == tmp_path / "qvl_clients.config.json"

    def test_get_machine_config_empty(self, tmp_path):
        config = ExplorationConfig(config_dir=tmp_path)
        result = config.get_machine_config(
            endpoints=["http://host1:29800"],
            endpoint="http://host1:29800",
        )
        assert result is None

    def test_save_machine_config(self, tmp_path):
        config = ExplorationConfig(config_dir=tmp_path)
        config.save_machine_config(
            endpoints=["http://host1:29800"],
            endpoint="http://host1:29800",
            optimal_max_concurrent=16,
            throughput=5.2,
            instances=4,
        )
        result = config.get_machine_config(
            endpoints=["http://host1:29800"],
            endpoint="http://host1:29800",
        )
        assert result is not None
        assert result["optimal_max_concurrent"] == 16
        assert result["throughput"] == 5.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
