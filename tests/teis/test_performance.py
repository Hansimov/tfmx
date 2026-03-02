"""Tests for tfmx.teis.performance module"""

import pytest
import json

from pathlib import Path
from tfmx.teis.performance import ExplorationConfig


class TestExplorationConfig:
    """Test ExplorationConfig persistence"""

    def test_init_no_file(self, tmp_path):
        """Init with no existing config file"""
        config = ExplorationConfig(config_dir=tmp_path)
        assert config._config == {}

    def test_save_and_load(self, tmp_path):
        """Save and reload config"""
        config1 = ExplorationConfig(config_dir=tmp_path)
        # Manually set some data
        config1._config = {"test_key": {"batch_size": 100}}
        config1._save()

        config2 = ExplorationConfig(config_dir=tmp_path)
        assert "test_key" in config2._config
        assert config2._config["test_key"]["batch_size"] == 100

    def test_config_file_location(self, tmp_path):
        """Config file should be in the specified directory"""
        config = ExplorationConfig(config_dir=tmp_path)
        assert config.config_path == tmp_path / "tei_clients.config.json"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
