#!/usr/bin/env python3
"""
Analyze memory efficiency between single and batch operations.
This script helps identify zero-copy violations in batch implementations.
"""

import subprocess
import json
import sys
from typing import Dict, List, Tuple

def run_memory_test() -> Dict[str, Tuple[int, int]]:
    """Run the memory efficiency test and parse results."""
    print("Running memory efficiency test...")

    try:
        result = subprocess.run(
            ["cargo", "test", "--bin", "memory_efficiency_test", "--release"],
            capture_output=True,
            text=True,
            check=True
        )


        lines = result.stdout.split('\n')
        results = {}

        for line in lines:

            if " - Single: " in line and ", Batch[1]: " in line:
                parts = line.split(" - ")
                if len(parts) == 2:
                    indicator = parts[0].strip()
                    memory_parts = parts[1].split(", ")

                    single_str = memory_parts[0].replace("Single: ", "").strip()
                    batch_str = memory_parts[1].replace("Batch[1]: ", "").strip()


                    single_bytes = parse_bytes(single_str)
                    batch_bytes = parse_bytes(batch_str)

                    results[indicator] = (single_bytes, batch_bytes)

        return results

    except subprocess.CalledProcessError as e:
        print(f"Error running memory test: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)

def parse_bytes(size_str: str) -> int:
    """Parse size string like '78.12 KB' to bytes."""
    parts = size_str.split()
    if len(parts) != 2:
        return 0

    value = float(parts[0])
    unit = parts[1].upper()

    if unit == 'B':
        return int(value)
    elif unit == 'KB':
        return int(value * 1024)
    elif unit == 'MB':
        return int(value * 1024 * 1024)
    else:
        return 0

def analyze_results(results: Dict[str, Tuple[int, int]]) -> None:
    """Analyze and report on memory efficiency."""
    print("\n=== Memory Efficiency Analysis ===\n")

    issues = []
    warnings = []
    good = []

    for indicator, (single_mem, batch_mem) in sorted(results.items()):
        if single_mem == 0:
            continue

        overhead_ratio = batch_mem / single_mem
        overhead_pct = (overhead_ratio - 1) * 100


        if overhead_pct > 50:
            issues.append((indicator, single_mem, batch_mem, overhead_pct))
        elif overhead_pct > 10:
            warnings.append((indicator, single_mem, batch_mem, overhead_pct))
        else:
            good.append((indicator, single_mem, batch_mem, overhead_pct))


    if issues:
        print("❌ HIGH OVERHEAD (>50%) - Likely zero-copy violations:")
        print("-" * 60)
        for ind, single, batch, overhead in issues:
            print(f"  {ind:<10} Single: {format_bytes(single):<10} "
                  f"Batch[1]: {format_bytes(batch):<10} "
                  f"Overhead: {overhead:>5.0f}%")
        print()


    if warnings:
        print("⚠️  MODERATE OVERHEAD (10-50%) - May need optimization:")
        print("-" * 60)
        for ind, single, batch, overhead in warnings:
            print(f"  {ind:<10} Single: {format_bytes(single):<10} "
                  f"Batch[1]: {format_bytes(batch):<10} "
                  f"Overhead: {overhead:>5.0f}%")
        print()


    if good:
        print("✅ EFFICIENT (<10% overhead) - Good implementations:")
        print("-" * 60)
        for ind, single, batch, overhead in good:
            print(f"  {ind:<10} Single: {format_bytes(single):<10} "
                  f"Batch[1]: {format_bytes(batch):<10} "
                  f"Overhead: {overhead:>5.0f}%")
        print()


    print("\n=== Summary ===")
    print(f"Total indicators tested: {len(results)}")
    print(f"High overhead issues: {len(issues)}")
    print(f"Moderate overhead warnings: {len(warnings)}")
    print(f"Efficient implementations: {len(good)}")

    if issues:
        print("\n⚠️  Recommendations:")
        print("1. Review batch implementations with high overhead")
        print("2. Check for unnecessary data copying in batch operations")
        print("3. Ensure batch operations reuse input data references")
        print("4. Consider using views/slices instead of cloning data")

def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.1f} KB"
    else:
        return f"{bytes_val / (1024 * 1024):.1f} MB"

def run_benchmarks() -> None:
    """Run the simplified benchmarks."""
    print("\n=== Running Performance Benchmarks ===\n")

    try:

        subprocess.run(
            ["cargo", "bench", "--bench", "benchmark_simple"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmarks: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    print("Memory Efficiency Analysis Tool")
    print("=" * 40)


    print("\nBuilding memory efficiency test...")
    try:
        subprocess.run(
            ["cargo", "build", "--bin", "memory_efficiency_test", "--release"],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError:
        print("Error: Could not build memory_efficiency_test")
        print("Make sure benches/memory_efficiency_test.rs is properly configured")
        sys.exit(1)


    results = run_memory_test()
    if results:
        analyze_results(results)
    else:
        print("Error: No results from memory test")
        sys.exit(1)


    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        run_benchmarks()

if __name__ == "__main__":
    main()