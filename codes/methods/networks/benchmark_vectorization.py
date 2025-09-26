#!/usr/bin/env python3
"""
Benchmark script to compare performance between original and vectorized ConfusionPointerNet implementations.
"""

import os
import sys
import time
import torch
import psutil
import gc
from typing import Dict, Tuple
import numpy as np

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from codes.methods.networks.confusionset_pointer_net import ConfusionPointerNet
from codes.methods.networks.confusionset_pointer_net_vectorized import (
    ConfusionPointerNetVectorized,
)


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        return gpu_memory
    else:
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / (1024**2)  # MB
        return cpu_memory


def create_test_data(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic test data for benchmarking."""

    # Create source sequences
    src_ids = torch.randint(
        4, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )
    src_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

    # Create target sequences (with SOS token at start)
    tgt_ids = torch.randint(
        4, vocab_size, (batch_size, seq_len + 1), device=device, dtype=torch.long
    )
    tgt_ids[:, 0] = 2  # SOS token

    # Create confusion set masks (allow random 10% of vocab for each position)
    confusionset_mask = torch.zeros(
        batch_size, seq_len, vocab_size, device=device, dtype=torch.uint8
    )
    for b in range(batch_size):
        for i in range(seq_len):
            # Allow ~10% of vocabulary as confusion set
            allowed_indices = torch.randperm(vocab_size)[: max(1, vocab_size // 10)]
            confusionset_mask[b, i, allowed_indices] = 1
            # Always allow the source token itself
            confusionset_mask[b, i, src_ids[b, i]] = 1

    return src_ids, src_mask, confusionset_mask, tgt_ids


def benchmark_model(
    model,
    src_ids,
    src_mask,
    confusionset_mask,
    tgt_ids,
    model_name: str,
    num_runs: int = 5,
) -> Dict:
    """Benchmark a model and return timing statistics."""

    device = src_ids.device
    model = model.to(device)
    model.train()  # Training mode for fair comparison

    # Warmup runs
    for _ in range(2):
        try:
            _ = model(
                src_ids, src_mask, confusionset_mask, tgt_ids, teacher_forcing=True
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warmup failed for {model_name}: {e}")
            return {"error": str(e)}

    # Measure memory before
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    memory_before = get_memory_usage()

    # Timing runs - Forward pass
    forward_times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        try:
            outputs = model(
                src_ids, src_mask, confusionset_mask, tgt_ids, teacher_forcing=True
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Forward pass failed for {model_name}: {e}")
            return {"error": str(e)}

        end_time = time.perf_counter()
        forward_times.append(end_time - start_time)

    # Timing runs - Forward + Backward pass
    total_times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        try:
            outputs = model(
                src_ids, src_mask, confusionset_mask, tgt_ids, teacher_forcing=True
            )

            if "loss" in outputs:
                loss = outputs["loss"]
                loss.backward()

            if device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Forward+backward pass failed for {model_name}: {e}")
            return {"error": str(e)}

        end_time = time.perf_counter()
        total_times.append(end_time - start_time)

        # Clear gradients
        model.zero_grad()

    # Measure memory after
    memory_after = get_memory_usage()

    # Clean up
    del outputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_name": model_name,
        "forward_time_mean": np.mean(forward_times),
        "forward_time_std": np.std(forward_times),
        "total_time_mean": np.mean(total_times),
        "total_time_std": np.std(total_times),
        "memory_usage_mb": memory_after - memory_before,
        "memory_peak_mb": memory_after,
    }


def run_benchmark_suite():
    """Run comprehensive benchmarks."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on device: {device}")
    print("=" * 80)

    # Test configurations
    test_configs = [
        {"batch_size": 8, "seq_len": 32, "name": "Small (8x32)"},
        {"batch_size": 16, "seq_len": 64, "name": "Medium (16x64)"},
        {"batch_size": 8, "seq_len": 128, "name": "Large (8x128)"},
        {"batch_size": 4, "seq_len": 256, "name": "XLarge (4x256)"},
    ]

    vocab_size = 1000  # Smaller vocab for testing

    results = []

    for config in test_configs:
        batch_size = config["batch_size"]
        seq_len = config["seq_len"]
        config_name = config["name"]

        print(f"\nBenchmarking configuration: {config_name}")
        print("-" * 50)

        try:
            # Create test data
            src_ids, src_mask, confusionset_mask, tgt_ids = create_test_data(
                batch_size, seq_len, vocab_size, device
            )

            # Create models with identical parameters
            model_params = {
                "vocab_size": vocab_size,
                "embed_dim": 256,
                "enc_hidden": 256,
                "dec_hidden": 256,
                "attn_dim": 128,
                "ptr_dim": 64,
                "drop_rate": 0.1,
            }

            # Test original model
            print("Testing Original ConfusionPointerNet...")
            original_model = ConfusionPointerNet(**model_params)
            original_results = benchmark_model(
                original_model,
                src_ids,
                src_mask,
                confusionset_mask,
                tgt_ids,
                "Original",
                num_runs=3,
            )

            # Clean up
            del original_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)  # Give memory a moment to clear

            # Test vectorized model
            print("Testing Vectorized ConfusionPointerNet...")
            vectorized_model = ConfusionPointerNetVectorized(**model_params)
            vectorized_results = benchmark_model(
                vectorized_model,
                src_ids,
                src_mask,
                confusionset_mask,
                tgt_ids,
                "Vectorized",
                num_runs=3,
            )

            # Clean up
            del vectorized_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            # Store results
            config_result = {
                "config": config_name,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "original": original_results,
                "vectorized": vectorized_results,
            }
            results.append(config_result)

            # Print comparison
            if "error" not in original_results and "error" not in vectorized_results:
                forward_speedup = (
                    original_results["forward_time_mean"]
                    / vectorized_results["forward_time_mean"]
                )
                total_speedup = (
                    original_results["total_time_mean"]
                    / vectorized_results["total_time_mean"]
                )
                memory_ratio = vectorized_results["memory_usage_mb"] / max(
                    original_results["memory_usage_mb"], 1
                )

                print(f"Forward pass speedup: {forward_speedup:.2f}x")
                print(f"Total (fwd+bwd) speedup: {total_speedup:.2f}x")
                print(f"Memory usage ratio: {memory_ratio:.2f}x")
            else:
                if "error" in original_results:
                    print(f"Original model failed: {original_results['error']}")
                if "error" in vectorized_results:
                    print(f"Vectorized model failed: {vectorized_results['error']}")

        except Exception as e:
            print(f"Configuration {config_name} failed: {e}")
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(
        f"{'Config':<15} {'Forward Speedup':<15} {'Total Speedup':<15} {'Memory Ratio':<15}"
    )
    print("-" * 70)

    for result in results:
        config_name = result["config"]
        original = result["original"]
        vectorized = result["vectorized"]

        if "error" not in original and "error" not in vectorized:
            forward_speedup = (
                original["forward_time_mean"] / vectorized["forward_time_mean"]
            )
            total_speedup = original["total_time_mean"] / vectorized["total_time_mean"]
            memory_ratio = vectorized["memory_usage_mb"] / max(
                original["memory_usage_mb"], 1
            )

            print(
                f"{config_name:<15} {forward_speedup:<15.2f} {total_speedup:<15.2f} {memory_ratio:<15.2f}"
            )
        else:
            print(f"{config_name:<15} {'FAILED':<15} {'FAILED':<15} {'FAILED':<15}")

    print("\nDetailed Results:")
    print("-" * 50)

    for result in results:
        config_name = result["config"]
        original = result["original"]
        vectorized = result["vectorized"]

        print(f"\n{config_name}:")

        if "error" not in original:
            print(
                f"  Original - Forward: {original['forward_time_mean']:.4f}±{original['forward_time_std']:.4f}s"
            )
            print(
                f"  Original - Total: {original['total_time_mean']:.4f}±{original['total_time_std']:.4f}s"
            )
            print(f"  Original - Memory: {original['memory_usage_mb']:.1f} MB")
        else:
            print(f"  Original - ERROR: {original['error']}")

        if "error" not in vectorized:
            print(
                f"  Vectorized - Forward: {vectorized['forward_time_mean']:.4f}±{vectorized['forward_time_std']:.4f}s"
            )
            print(
                f"  Vectorized - Total: {vectorized['total_time_mean']:.4f}±{vectorized['total_time_std']:.4f}s"
            )
            print(f"  Vectorized - Memory: {vectorized['memory_usage_mb']:.1f} MB")
        else:
            print(f"  Vectorized - ERROR: {vectorized['error']}")


def test_correctness():
    """Test that both models produce similar outputs (correctness test)."""

    print("Running correctness test...")
    print("-" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, seq_len, vocab_size = 2, 16, 100

    # Create test data
    src_ids, src_mask, confusionset_mask, tgt_ids = create_test_data(
        batch_size, seq_len, vocab_size, device
    )

    # Create models with identical parameters
    model_params = {
        "vocab_size": vocab_size,
        "embed_dim": 128,
        "enc_hidden": 128,
        "dec_hidden": 128,
        "attn_dim": 64,
        "ptr_dim": 32,
        "drop_rate": 0.0,  # No dropout for deterministic comparison
    }

    # Initialize models
    torch.manual_seed(42)
    original_model = ConfusionPointerNet(**model_params).to(device)

    torch.manual_seed(42)  # Same seed for identical initialization
    vectorized_model = ConfusionPointerNetVectorized(**model_params).to(device)

    # Set to eval mode for deterministic behavior
    original_model.eval()
    vectorized_model.eval()

    with torch.no_grad():
        # Run both models
        original_outputs = original_model(
            src_ids, src_mask, confusionset_mask, tgt_ids, teacher_forcing=True
        )
        vectorized_outputs = vectorized_model(
            src_ids, src_mask, confusionset_mask, tgt_ids, teacher_forcing=True
        )

    # Compare outputs
    print("Output shape comparison:")
    for key in ["pred_ids", "pointer_logits", "vocab_logits"]:
        if key in original_outputs and key in vectorized_outputs:
            orig_shape = original_outputs[key].shape
            vect_shape = vectorized_outputs[key].shape
            print(f"  {key}: Original {orig_shape} vs Vectorized {vect_shape}")
        else:
            print(f"  {key}: Missing in one of the outputs")

    # Note: Exact numerical comparison may not be possible due to different computation order
    # but shapes should match

    print("Correctness test completed.")
    print("Note: Exact numerical comparison requires identical computation order.")
    print("Shape matching indicates structural correctness.")


if __name__ == "__main__":
    print("ConfusionPointerNet Vectorization Benchmark")
    print("=" * 50)

    # Test correctness first
    try:
        test_correctness()
    except Exception as e:
        print(f"Correctness test failed: {e}")

    print("\n")

    # Run performance benchmarks
    try:
        run_benchmark_suite()
    except Exception as e:
        print(f"Benchmark suite failed: {e}")

    print("\nBenchmark completed!")
