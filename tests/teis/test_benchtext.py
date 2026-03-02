"""Tests for tfmx.teis.benchtext module"""

import pytest
import random

from tfmx.teis.benchtext import TEIBenchTextGenerator


class TestTEIBenchTextGenerator:
    """Test text generation for benchmarking"""

    def test_init(self):
        gen = TEIBenchTextGenerator(seed=42)
        assert gen is not None

    def test_generate_one(self):
        gen = TEIBenchTextGenerator(seed=42)
        text = gen._generate_one(min_len=10, max_len=100)
        assert isinstance(text, str)
        assert len(text) >= 10

    def test_generate_batch(self):
        gen = TEIBenchTextGenerator(seed=42)
        texts = gen.generate(count=50, min_len=10, max_len=200, show_progress=False)
        assert len(texts) == 50
        assert all(isinstance(t, str) for t in texts)
        assert all(len(t) >= 10 for t in texts)

    def test_deterministic_with_seed(self):
        """Same seed produces same texts"""
        gen1 = TEIBenchTextGenerator(seed=42)
        gen2 = TEIBenchTextGenerator(seed=42)
        texts1 = gen1.generate(count=10, min_len=10, max_len=100, show_progress=False)
        texts2 = gen2.generate(count=10, min_len=10, max_len=100, show_progress=False)
        assert texts1 == texts2

    def test_different_seeds_different_texts(self):
        gen1 = TEIBenchTextGenerator(seed=1)
        gen2 = TEIBenchTextGenerator(seed=2)
        texts1 = gen1.generate(count=10, min_len=10, max_len=100, show_progress=False)
        texts2 = gen2.generate(count=10, min_len=10, max_len=100, show_progress=False)
        assert texts1 != texts2

    def test_diverse_output(self):
        """Generated texts should not all be identical"""
        gen = TEIBenchTextGenerator(seed=42)
        texts = gen.generate(count=20, min_len=10, max_len=200, show_progress=False)
        unique_texts = set(texts)
        assert len(unique_texts) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
