"""Tests for tfmx.utils.lsh module"""

import pytest
import numpy as np

from tfmx.utils.lsh import LSHConverter


class TestLSHConverterInit:
    """Test LSHConverter initialization"""

    def test_default_init(self):
        """Default initialization with CPU"""
        lsh = LSHConverter(dims=64, bitn=128, use_gpu=False)
        assert lsh.dims == 64
        assert lsh.bitn == 128
        assert lsh.seed == 1
        assert lsh.hps is not None
        assert lsh.hps.shape == (128, 64)

    def test_custom_seed(self):
        """Different seeds produce different hyperplanes"""
        lsh1 = LSHConverter(dims=64, bitn=128, seed=1, use_gpu=False)
        lsh2 = LSHConverter(dims=64, bitn=128, seed=2, use_gpu=False)
        assert not np.allclose(lsh1.hps, lsh2.hps)

    def test_same_seed_reproducible(self):
        """Same seed produces identical hyperplanes"""
        lsh1 = LSHConverter(dims=64, bitn=128, seed=42, use_gpu=False)
        lsh2 = LSHConverter(dims=64, bitn=128, seed=42, use_gpu=False)
        np.testing.assert_array_equal(lsh1.hps, lsh2.hps)

    def test_hyperplanes_normalized(self):
        """Hyperplanes should be unit vectors"""
        lsh = LSHConverter(dims=64, bitn=128, seed=1, use_gpu=False)
        norms = np.linalg.norm(lsh.hps, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestEmbsToBits:
    """Test embedding to bits conversion"""

    def test_single_embedding(self):
        """Single embedding input"""
        lsh = LSHConverter(dims=64, bitn=128, use_gpu=False)
        emb = np.random.randn(64).astype(np.float32)
        bits = lsh.embs_to_bits(emb)
        assert bits.shape == (128,)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_batch_embedding(self):
        """Batch of embeddings"""
        lsh = LSHConverter(dims=64, bitn=128, use_gpu=False)
        embs = np.random.randn(10, 64).astype(np.float32)
        bits = lsh.embs_to_bits(embs)
        assert bits.shape == (10, 128)
        assert set(np.unique(bits)).issubset({0, 1})

    def test_deterministic(self):
        """Same input produces same output"""
        lsh = LSHConverter(dims=64, bitn=128, seed=1, use_gpu=False)
        emb = np.ones(64, dtype=np.float32)
        bits1 = lsh.embs_to_bits(emb)
        bits2 = lsh.embs_to_bits(emb)
        np.testing.assert_array_equal(bits1, bits2)


class TestBitsToHex:
    """Test bits to hex conversion"""

    def test_basic_conversion(self):
        """Known bits -> hex"""
        lsh = LSHConverter(dims=64, bitn=8, use_gpu=False)
        bits = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        hex_str = lsh.bits_to_hex(bits)
        assert hex_str == "f0"

    def test_all_zeros(self):
        lsh = LSHConverter(dims=64, bitn=8, use_gpu=False)
        bits = np.zeros(8, dtype=np.uint8)
        hex_str = lsh.bits_to_hex(bits)
        assert hex_str == "00"

    def test_all_ones(self):
        lsh = LSHConverter(dims=64, bitn=8, use_gpu=False)
        bits = np.ones(8, dtype=np.uint8)
        hex_str = lsh.bits_to_hex(bits)
        assert hex_str == "ff"


class TestEmbsToHexBatch:
    """Test batch embedding to hex conversion (CPU path)"""

    def test_single(self):
        """Single embedding in batch"""
        lsh = LSHConverter(dims=64, bitn=128, use_gpu=False)
        embs = np.random.randn(1, 64).astype(np.float32)
        hexes = lsh.embs_to_hex_batch(embs)
        assert len(hexes) == 1
        assert isinstance(hexes[0], str)
        assert len(hexes[0]) == 128 // 4  # 128 bits = 32 hex chars

    def test_batch(self):
        """Multiple embeddings"""
        lsh = LSHConverter(dims=64, bitn=128, use_gpu=False)
        embs = np.random.randn(100, 64).astype(np.float32)
        hexes = lsh.embs_to_hex_batch(embs)
        assert len(hexes) == 100
        assert all(isinstance(h, str) for h in hexes)

    def test_1d_input(self):
        """1D input gets reshaped"""
        lsh = LSHConverter(dims=64, bitn=128, use_gpu=False)
        emb = np.random.randn(64).astype(np.float32)
        hexes = lsh.embs_to_hex_batch(emb)
        assert len(hexes) == 1

    def test_deterministic(self):
        """Same input -> same output"""
        lsh = LSHConverter(dims=64, bitn=128, seed=1, use_gpu=False)
        embs = np.ones((5, 64), dtype=np.float32)
        hexes1 = lsh.embs_to_hex_batch(embs)
        hexes2 = lsh.embs_to_hex_batch(embs)
        assert hexes1 == hexes2

    def test_identical_embeddings_same_hash(self):
        """Identical embeddings produce identical hashes"""
        lsh = LSHConverter(dims=64, bitn=128, seed=1, use_gpu=False)
        emb = np.random.randn(64).astype(np.float32)
        embs = np.stack([emb, emb, emb])
        hexes = lsh.embs_to_hex_batch(embs)
        assert hexes[0] == hexes[1] == hexes[2]

    def test_similar_embeddings_similar_hashes(self):
        """Similar embeddings should have similar hashes (LSH property)"""
        lsh = LSHConverter(dims=64, bitn=256, seed=1, use_gpu=False)
        emb1 = np.random.randn(64).astype(np.float32)
        emb2 = emb1 + np.random.randn(64).astype(np.float32) * 0.01  # tiny perturbation
        emb3 = np.random.randn(64).astype(np.float32)  # completely different

        hexes = lsh.embs_to_hex_batch(np.stack([emb1, emb2, emb3]))

        # Convert hex to bits for hamming distance
        def hex_to_hamming(h1, h2):
            b1 = bytes.fromhex(h1)
            b2 = bytes.fromhex(h2)
            return sum((x ^ y).bit_count() for x, y in zip(b1, b2))

        dist_similar = hex_to_hamming(hexes[0], hexes[1])
        dist_different = hex_to_hamming(hexes[0], hexes[2])

        # Similar embeddings should have smaller hamming distance
        assert dist_similar < dist_different


class TestLSHConverterSaveLoad:
    """Test hyperplane save/load"""

    def test_save_and_load(self, tmp_path):
        """Save and reload hyperplanes"""
        lsh1 = LSHConverter(dims=32, bitn=64, seed=99, use_gpu=False)
        # Override save path
        save_path = tmp_path / "test_hps.npy"
        lsh1.hps_path = save_path
        lsh1.save_hyperplanes()

        assert save_path.exists()

        lsh2 = LSHConverter(dims=32, bitn=64, seed=99, use_gpu=False)
        lsh2.hps_path = save_path
        lsh2.load_hyperplanes()

        np.testing.assert_array_equal(lsh1.hps, lsh2.hps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
