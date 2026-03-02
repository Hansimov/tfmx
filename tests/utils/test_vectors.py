"""Tests for tfmx.utils.vectors module"""

import pytest
import numpy as np

from tfmx.utils.vectors import (
    floats_to_bits,
    bits_to_hash,
    bits_dist,
    hash_dist,
    bits_sim,
    hash_sim,
    dot_sim,
)


class TestFloatsToBits:
    """Test floats_to_bits conversion"""

    def test_basic_k1(self):
        """k=1: each float maps to 1 bit based on midpoint"""
        floats = [-0.75, -0.23, 0.0, 0.24, 0.85]
        bits = floats_to_bits(floats, k=1)
        assert len(bits) == len(floats) * 1
        assert all(b in (0, 1) for b in bits)

    def test_basic_k2(self):
        """k=2: each float maps to 2 bits"""
        floats = [-0.75, -0.23, 0.0, 0.24, 0.85]
        bits = floats_to_bits(floats, k=2)
        assert len(bits) == len(floats) * 2
        assert all(b in (0, 1) for b in bits)

    def test_single_float(self):
        """Single float with explicit bounds (auto_min_max would cause div-by-zero)"""
        bits = floats_to_bits([0.5], k=2, min_max=(0.0, 1.0))
        assert len(bits) == 2

    def test_custom_min_max(self):
        """Custom bounds"""
        bits = floats_to_bits([0.0, 1.0], k=1, min_max=(0.0, 1.0))
        assert len(bits) == 2
        assert bits[0] == 0  # min maps to 0
        assert bits[1] == 1  # max maps to 1

    def test_auto_min_max_disabled(self):
        """Uses default [-1, 1] when auto_min_max=False"""
        # With auto_min_max=True on multi-element list
        bits1 = floats_to_bits([0.5, -0.5], k=2, auto_min_max=True)
        bits2 = floats_to_bits([0.5, -0.5], k=2, auto_min_max=False)
        assert len(bits1) == len(bits2) == 4

    def test_empty_list(self):
        """Empty input raises ValueError (min() on empty)"""
        with pytest.raises(ValueError):
            floats_to_bits([], k=2)

    def test_higher_k(self):
        """k=4: 4 bits per float"""
        floats = [0.0, 1.0]
        bits = floats_to_bits(floats, k=4)
        assert len(bits) == 8


class TestBitsToHash:
    """Test bits_to_hash conversion"""

    def test_basic(self):
        """Known conversion"""
        bits = [1, 1, 1, 1, 0, 0, 0, 0]
        h = bits_to_hash(bits)
        assert h == "f0"

    def test_all_zeros(self):
        """All zeros"""
        bits = [0, 0, 0, 0, 0, 0, 0, 0]
        h = bits_to_hash(bits)
        assert h == "00"

    def test_all_ones(self):
        """All ones"""
        bits = [1, 1, 1, 1, 1, 1, 1, 1]
        h = bits_to_hash(bits)
        assert h == "ff"

    def test_empty(self):
        """Empty bits"""
        h = bits_to_hash([])
        assert h == ""

    def test_non_8_aligned(self):
        """Non-8-aligned bits get padded"""
        bits = [1, 0, 1]
        h = bits_to_hash(bits)
        # 101 -> 10100000 -> a0
        assert h == "a0"


class TestBitsDist:
    """Test hamming distance between bit vectors"""

    def test_identical(self):
        assert bits_dist([0, 1, 0, 1], [0, 1, 0, 1]) == 0

    def test_all_different(self):
        assert bits_dist([0, 0, 0, 0], [1, 1, 1, 1]) == 4

    def test_one_different(self):
        assert bits_dist([0, 0, 0, 0], [0, 0, 0, 1]) == 1

    def test_empty(self):
        assert bits_dist([], []) == 0


class TestHashDist:
    """Test hamming distance between hash strings"""

    def test_identical(self):
        assert hash_dist("ff", "ff") == 0

    def test_one_bit_diff(self):
        # fe = 11111110, ff = 11111111 -> 1 bit difference
        assert hash_dist("fe", "ff") == 1

    def test_all_different(self):
        # 00 = 00000000, ff = 11111111 -> 8 bits
        assert hash_dist("00", "ff") == 8


class TestBitsSim:
    """Test bit similarity"""

    def test_identical(self):
        assert bits_sim([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0

    def test_all_different(self):
        assert bits_sim([0, 0, 0, 0], [1, 1, 1, 1]) == 0.0

    def test_half_match(self):
        assert bits_sim([0, 0, 1, 1], [0, 0, 0, 0]) == 0.5

    def test_rounding(self):
        sim = bits_sim([0, 0, 0], [0, 0, 1], ndigits=2)
        assert sim == round(2.0 / 3.0, 2)


class TestHashSim:
    """Test hash similarity"""

    def test_identical(self):
        assert hash_sim("ff", "ff") == 1.0

    def test_all_different(self):
        assert hash_sim("00", "ff") == 0.0


class TestDotSim:
    """Test dot product similarity"""

    def test_identical(self):
        v = np.array([1.0, 0.0, 0.0])
        assert dot_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert dot_sim(v1, v2) == pytest.approx(0.0)

    def test_opposite(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert dot_sim(v1, v2) == pytest.approx(-1.0)

    def test_rounding(self):
        v1 = np.array([1.0, 1.0, 1.0])
        v2 = np.array([1.0, 0.0, 0.0])
        sim = dot_sim(v1, v2, ndigits=4)
        expected = round(1.0 / np.sqrt(3), 4)
        assert sim == expected


class TestRoundTrip:
    """Test floats -> bits -> hash -> distance/similarity chain"""

    def test_roundtrip_consistency(self):
        """The full chain should produce consistent results"""
        floats1 = [0.1, 0.5, -0.3, 0.8]
        floats2 = [0.1, 0.5, -0.3, 0.8]
        bits1 = floats_to_bits(floats1, k=2)
        bits2 = floats_to_bits(floats2, k=2)
        hash1 = bits_to_hash(bits1)
        hash2 = bits_to_hash(bits2)
        assert bits_dist(bits1, bits2) == 0
        assert hash_dist(hash1, hash2) == 0
        assert bits_sim(bits1, bits2) == 1.0
        assert hash_sim(hash1, hash2) == 1.0

    def test_different_floats_produce_different_hashes(self):
        floats1 = [0.1, 0.5, -0.3, 0.8]
        floats2 = [-0.9, -0.5, 0.7, -0.1]
        bits1 = floats_to_bits(floats1, k=2)
        bits2 = floats_to_bits(floats2, k=2)
        hash1 = bits_to_hash(bits1)
        hash2 = bits_to_hash(bits2)
        assert hash1 != hash2
        assert bits_dist(bits1, bits2) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
