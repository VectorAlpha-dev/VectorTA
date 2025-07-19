#!/usr/bin/env python3
"""Final test to verify tilson optimization achieves <10% overhead."""

import time
import numpy as np
import subprocess
import sys
import json

def run_rust_benchmark():
    """Run Rust benchmark and extract median time."""
    print("Running Rust benchmark for tilson...")
    try:
        # Run cargo bench for tilson
        result = subprocess.run(
            ["cargo", "bench", "--features", "nightly-avx", "--bench", "indicator_benchmark", "--", "tilson", "1M"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Parse output for median time
        output = result.stdout
        # Look for pattern like "tilson/1M      time:   [X.XXX ms Y.YYY ms Z.ZZZ ms]"
        # Where Y.YYY is the median
        import re
        pattern = r"tilson/1M\s+time:\s+\[[\d.]+ ms\s+([\d.]+) ms"
        match = re.search(pattern, output)
        
        if match:
            return float(match.group(1))
        else:
            print("Could not parse Rust benchmark output")
            print("Output:", output[:500])
            return None
    except Exception as e:
        print(f"Error running Rust benchmark: {e}")
        return None

def run_python_benchmark():
    """Run Python benchmark for tilson."""
    print("\nRunning Python benchmark for tilson...")
    try:
        import my_project
        
        # Load test data
        np.random.seed(42)
        data = np.random.randn(1_000_000).astype(np.float64)
        
        # Warmup
        print("  Warming up...")
        for _ in range(10):
            _ = my_project.tilson(data, 14, 0.7)
        
        # Benchmark
        print("  Running benchmark...")
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = my_project.tilson(data, 14, 0.7)
            times.append((time.perf_counter() - start) * 1000)  # Convert to ms
        
        return np.median(times)
    except Exception as e:
        print(f"Error running Python benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Tilson Optimization Verification")
    print("=" * 50)
    
    # Run benchmarks
    rust_time = run_rust_benchmark()
    python_time = run_python_benchmark()
    
    if rust_time is None or python_time is None:
        print("\nError: Could not complete benchmarks")
        return 1
    
    # Calculate overhead
    overhead_ms = python_time - rust_time
    overhead_pct = (overhead_ms / rust_time) * 100
    
    print("\nResults:")
    print(f"  Rust median time:   {rust_time:.3f} ms")
    print(f"  Python median time: {python_time:.3f} ms")
    print(f"  Overhead:           {overhead_ms:.3f} ms ({overhead_pct:.1f}%)")
    
    # Check if optimization target is met
    if overhead_pct < 10:
        print(f"\n✓ SUCCESS: Python binding overhead is {overhead_pct:.1f}% (target: <10%)")
        return 0
    else:
        print(f"\n✗ FAILED: Python binding overhead is {overhead_pct:.1f}% (target: <10%)")
        return 1

if __name__ == "__main__":
    sys.exit(main())