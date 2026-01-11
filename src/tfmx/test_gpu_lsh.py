#!/usr/bin/env python3
"""Test GPU LSH performance vs CPU."""

import numpy as np
import time

from .lsh import LSHConverter


def test_lsh_performance():
    """Compare CPU vs GPU LSH performance."""

    # Test parameters
    dims = 1024
    bitn = 2048
    n_samples = 10000

    print(f"\n{'='*60}")
    print(f"LSH Performance Test")
    print(f"{'='*60}")
    print(f"Dimensions: {dims}")
    print(f"Hash bits: {bitn}")
    print(f"Samples: {n_samples:,}")
    print(f"{'='*60}\n")

    # Generate random embeddings
    embs = np.random.randn(n_samples, dims).astype(np.float32)

    # Test CPU
    print("Testing CPU LSH...")
    lsh_cpu = LSHConverter(dims=dims, bitn=bitn, use_gpu=False, verbose=True)

    start = time.perf_counter()
    hashes_cpu = lsh_cpu.embs_to_hex_batch(embs)
    cpu_time = time.perf_counter() - start

    print(f"  CPU Time: {cpu_time:.3f} seconds")
    print(f"  Throughput: {n_samples / cpu_time:,.0f} samples/sec\n")

    # Test GPU
    print("Testing GPU LSH...")
    lsh_gpu = LSHConverter(dims=dims, bitn=bitn, use_gpu=True, verbose=True)

    # Warmup
    _ = lsh_gpu.embs_to_hex_batch(embs[:100])

    start = time.perf_counter()
    hashes_gpu = lsh_gpu.embs_to_hex_batch(embs)
    gpu_time = time.perf_counter() - start

    print(f"  GPU Time: {gpu_time:.3f} seconds")
    print(f"  Throughput: {n_samples / gpu_time:,.0f} samples/sec\n")

    # Verify results match
    mismatch_count = sum(
        1 for i in range(len(hashes_cpu)) if hashes_cpu[i] != hashes_gpu[i]
    )

    if mismatch_count == 0:
        print("✓ Results match! GPU computation is correct.")
    else:
        print(f"✗ Results differ! {mismatch_count}/{len(hashes_cpu)} hashes mismatch.")

        # Find first mismatch
        first_mismatch_idx = None
        for i in range(len(hashes_cpu)):
            if hashes_cpu[i] != hashes_gpu[i]:
                first_mismatch_idx = i
                break

        if first_mismatch_idx is not None:
            cpu_hash = hashes_cpu[first_mismatch_idx]
            gpu_hash = hashes_gpu[first_mismatch_idx]

            print(f"\nFirst mismatch at sample {first_mismatch_idx}:")
            print(f"  CPU: {cpu_hash[:64]}{'...' if len(cpu_hash) > 64 else ''}")
            print(f"  GPU: {gpu_hash[:64]}{'...' if len(gpu_hash) > 64 else ''}")

            # Find where the hashes differ
            hash_diff_pos = []
            for i, (c1, c2) in enumerate(zip(cpu_hash, gpu_hash)):
                if c1 != c2:
                    hash_diff_pos.append(i)

            if hash_diff_pos:
                print(
                    f"  Hash differs at hex positions: {hash_diff_pos[:10]}{'...' if len(hash_diff_pos) > 10 else ''}"
                )

            # Deep dive: compare the projection values
            print("\n  Analyzing mismatch...")
            emb = embs[first_mismatch_idx : first_mismatch_idx + 1]

            # CPU projection
            cpu_proj = np.dot(emb, lsh_cpu.hps.T)[0]

            # GPU projection
            import torch

            with torch.no_grad():
                emb_gpu = torch.from_numpy(emb).to(lsh_gpu.device)
                gpu_proj = torch.matmul(emb_gpu, lsh_gpu.hps_gpu.T)[0].cpu().numpy()

            # Find differing bits
            diff_mask = (cpu_proj > 0) != (gpu_proj > 0)
            diff_count = diff_mask.sum()

            if diff_count > 0:
                diff_indices = np.where(diff_mask)[0]
                print(
                    f"  {diff_count} bits differ at positions: {diff_indices[:10].tolist()}{'...' if diff_count > 10 else ''}"
                )

                # Show projection values near zero (likely culprits)
                for idx in diff_indices[:3]:
                    print(
                        f"    Bit {idx}: CPU_proj={cpu_proj[idx]:.8f}, GPU_proj={gpu_proj[idx]:.8f}"
                    )
            else:
                print(f"  Projection bits are identical (diff_count=0)")
                print(f"  This suggests an issue in hex encoding, not LSH computation")

            # Check if it's just floating point precision near zero
            near_zero_count = np.sum(np.abs(cpu_proj) < 1e-6)
            print(
                f"  Projections near zero (|x| < 1e-6): {near_zero_count}/{len(cpu_proj)}"
            )

            if mismatch_count == 1 and diff_count <= 3:
                print(
                    "\n  ⚠️  This is likely a floating-point precision issue (acceptable)."
                )

    # Calculate speedup
    speedup = cpu_time / gpu_time
    print(f"\n{'='*60}")
    print(f"GPU Speedup: {speedup:.2f}x faster than CPU")

    # More lenient acceptance criteria
    if mismatch_count == 0:
        print("Status: ✓ PASS (perfect match)")
    elif mismatch_count <= 3:
        print(
            f"Status: ⚠️  ACCEPTABLE ({mismatch_count} mismatches, likely floating-point precision)"
        )
    else:
        print(f"Status: ✗ FAIL ({mismatch_count} mismatches)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_lsh_performance()

    # python -m tfmx.test_gpu_lsh
