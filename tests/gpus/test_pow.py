"""Tests for tfmx.gpus.pow module (pure utility functions)"""

import pytest

from tfmx.gpus.pow import (
    is_none_or_empty,
    is_str_and_all,
    parse_idx,
    parse_val,
    parse_persistence_mode,
    parse_power_limit,
    parse_val_by_op_key,
    NvidiaSmiParser,
)


class TestPowUtilFunctions:
    """Test pure utility functions in pow module"""

    def test_is_none_or_empty(self):
        assert is_none_or_empty(None) is True
        assert is_none_or_empty("") is True
        assert is_none_or_empty("   ") is True
        assert is_none_or_empty("abc") is False

    def test_is_str_and_all(self):
        assert is_str_and_all("all") is True
        assert is_str_and_all("ALL") is True
        assert is_str_and_all("0") is False

    def test_parse_idx(self):
        assert parse_idx(0) == 0
        assert parse_idx("3") == 3

    def test_parse_val(self):
        assert parse_val("100") == 100
        assert parse_val("200") == 200
        assert parse_val(None) is None
        assert parse_val("") is None

    def test_parse_persistence_mode(self):
        assert parse_persistence_mode("0") == 0
        assert parse_persistence_mode("1") == 1
        with pytest.raises(ValueError):
            parse_persistence_mode("2")  # invalid raises
        assert parse_persistence_mode(None) is None

    def test_parse_power_limit(self):
        assert parse_power_limit("200") == 200
        assert parse_power_limit("300") == 300
        with pytest.raises(ValueError):
            parse_power_limit("50")  # below MIN_POWER_LIMIT
        with pytest.raises(ValueError):
            parse_power_limit("600")  # above MAX_POWER_LIMIT
        assert parse_power_limit(None) is None

    def test_parse_val_by_op_key_persistence(self):
        assert parse_val_by_op_key("1", "persistence_mode") == 1

    def test_parse_val_by_op_key_power_limit(self):
        assert parse_val_by_op_key("290", "power_limit") == 290


class TestNvidiaSmiParser:
    """Test NvidiaSmiParser.key_idx_val_to_ops"""

    def setup_method(self):
        self.parser = NvidiaSmiParser()

    def test_single_gpu_single_val(self):
        """Parse '0:290'"""
        ops = self.parser.key_idx_val_to_ops("power_limit", "0:290")
        assert len(ops) == 1
        assert ops[0] == ("power_limit", "set", "0", "290")

    def test_multiple_gpus_same_val(self):
        """Parse '0,1:290'"""
        ops = self.parser.key_idx_val_to_ops("power_limit", "0,1:290")
        assert len(ops) == 2
        assert ops[0] == ("power_limit", "set", "0", "290")
        assert ops[1] == ("power_limit", "set", "1", "290")

    def test_multiple_groups(self):
        """Parse '0,1:290;2,3:280'"""
        ops = self.parser.key_idx_val_to_ops("power_limit", "0,1:290;2,3:280")
        assert len(ops) == 4

    def test_get_only(self):
        """Parse '0' (get operation, no value)"""
        ops = self.parser.key_idx_val_to_ops("power_limit", "0")
        assert len(ops) == 1
        assert ops[0][1] == "get"

    def test_all_gpus(self):
        """Parse 'all:290'"""
        ops = self.parser.key_idx_val_to_ops("power_limit", "all:290")
        assert len(ops) == 1
        assert ops[0][2] == "all"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
