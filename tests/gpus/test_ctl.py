"""Tests for tfmx.gpus.ctl module (pure utility functions)"""

import pytest

from tfmx.gpus.ctl import (
    is_none_or_empty,
    is_str_and_all,
    parse_idx,
)


class TestIsNoneOrEmpty:
    def test_none(self):
        assert is_none_or_empty(None) is True

    def test_empty_string(self):
        assert is_none_or_empty("") is True

    def test_whitespace(self):
        assert is_none_or_empty("   ") is True

    def test_non_empty(self):
        assert is_none_or_empty("hello") is False

    def test_zero_string(self):
        assert is_none_or_empty("0") is False


class TestIsStrAndAll:
    def test_all_lowercase(self):
        assert is_str_and_all("all") is True

    def test_all_uppercase(self):
        assert is_str_and_all("ALL") is True

    def test_a_prefix(self):
        assert is_str_and_all("a") is True

    def test_number(self):
        assert is_str_and_all("0") is False

    def test_integer_input(self):
        assert is_str_and_all(0) is False


class TestParseIdx:
    def test_integer(self):
        assert parse_idx(0) == 0

    def test_string_integer(self):
        assert parse_idx("3") == 3

    def test_negative(self):
        assert parse_idx("-1") == -1
