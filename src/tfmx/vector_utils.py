import numpy as np

from typing import Literal

BitsType = list[Literal[0, 1]]


def floats_to_bits(
    floats: list[float],
    k: int = 2,
    min_max: tuple[float, float] = None,
    auto_min_max: bool = True,
) -> BitsType:
    """
    Inputs:
        - k: 1 float map to k bits.
            For 1024-dim float-vector, if k is 2, it would become 2048-dim bit-vector.
        - min_max: if set, use as bounds;
        - auto_min_max:
            - if True, use min(floats) and max(floats) as bounds;
            - if False, use default bounds [-1, 1].
    Outputs:
        - bits: list of 0/1, dim is len(floats) * k
    Core idea:
        - determine mapping bounds (set or auto or default)
        - map floats to [0, 2^k - 1]
        - convert ints to bits
    """
    n = len(floats) * k
    bits = [0] * n
    levels = 2**k - 1
    # bounds: set, auto, default
    if min_max:
        min_val, max_val = min_max
    elif auto_min_max:
        min_val, max_val = min(floats), max(floats)
    else:
        min_val, max_val = -1.0, 1.0
    # map floats to bits
    for float_idx, f in enumerate(floats):
        # map [min_val, max_val] to [0, 2^k-1]
        f = max(min_val, min(max_val, float(f)))
        q = round((f - min_val) / (max_val - min_val) * levels)
        q = min(max(q, 0), levels)
        # MSB -> LSB
        for bit_idx, i in enumerate(range(k - 1, -1, -1)):
            idx = float_idx * k + bit_idx
            if (q >> i) & 1:
                bits[idx] = 1
            else:
                bits[idx] = 0
    return bits


def bits_to_hash(bits: BitsType) -> str:
    """Convert bit-vector to hash hex-string."""
    n = len(bits)
    if n == 0:
        return ""
    # convert bits to int
    val = 0
    for b in bits:
        val = (val << 1) | (int(b) & 1)
    # pad to 8x bits, as bytes
    pad = (-n) % 8
    val <<= pad
    # to big-endian bytes and to hex
    byte_len = (n + 7) // 8
    hex_str = val.to_bytes(byte_len, "big").hex()
    return hex_str


def bits_dist(bits1: BitsType, bits2: BitsType) -> int:
    """calc hamming distance of two bit-vectors"""
    dist = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
    return dist


def hash_dist(hash1: str, hash2: str) -> int:
    """calc hamming distance of two hash-strings"""
    b1 = bytes.fromhex(hash1)
    b2 = bytes.fromhex(hash2)
    dist = sum((x ^ y).bit_count() for x, y in zip(b1, b2))
    return dist


def bits_sim(bits1: BitsType, bits2: BitsType, ndigits: int = None) -> float:
    """calc normalized similarity of two bit-vectors"""
    dist = bits_dist(bits1, bits2)
    bits_len = max(len(bits1), len(bits2))
    sim = 1.0 - dist / bits_len
    if ndigits:
        sim = round(sim, ndigits)
    return sim


def hash_sim(hash1: str, hash2: str, ndigits: int = None) -> float:
    """calc normalized similarity of two hash-strings"""
    dist = hash_dist(hash1, hash2)
    bits_len = max(len(hash1), len(hash2)) * 4
    sim = 1.0 - dist / bits_len
    if ndigits:
        sim = round(sim, ndigits)
    return sim


def dot_sim(v1: np.ndarray, v2: np.ndarray, ndigits: int = None) -> float:
    """calc normalized dot product of two vectors"""
    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if ndigits:
        sim = round(sim, ndigits)
    return sim


def test_floats_to_bits():
    floats = [-0.75, -0.23, 0.0, 0.24, 0.85]
    bits_1x = floats_to_bits(floats, k=1)
    bits_2x = floats_to_bits(floats, k=2)
    hash_str = bits_to_hash(bits_2x)
    print("floats   :", floats)
    print("bits (1x):", bits_1x)
    print("bits (2x):", bits_2x)
    print("hash     :", hash_str)


def test_hash_sim():
    floats1 = [-0.75, -0.23, 0.0, 0.24, 0.85]
    floats2 = [0.25, 0.2, 0.85, -0.3, -0.7]
    bits1 = floats_to_bits(floats1, k=2)
    bits2 = floats_to_bits(floats2, k=2)
    hash1 = bits_to_hash(bits1)
    hash2 = bits_to_hash(bits2)
    bdist = bits_dist(bits1, bits2)
    hdist = hash_dist(hash1, hash2)
    bsim = bits_sim(bits1, bits2, ndigits=4)
    hsim = hash_sim(hash1, hash2, ndigits=4)
    print("floats1   :", floats1)
    print("floats2   :", floats2)
    print("bits1     :", bits1)
    print("bits2     :", bits2)
    print("hash1     :", hash1)
    print("hash2     :", hash2)
    print("bits_dist :", bdist)
    print("hash_dist :", hdist)
    print("bits_sim  :", bsim)
    print("hash_sim  :", hsim)


if __name__ == "__main__":
    test_floats_to_bits()
    test_hash_sim()

    # python -m tfmx.vector_utils
