#!/usr/bin/env python3
"""
Comprehensive benchmark suite for ALMA Python bindings.
Tests performance across various data sizes, kernel types, and operations.
"""

import numpy as np
import time
import psutil
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
from my_project import alma, alma_batch, AlmaStream

# Configuration
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10
DATA_SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]
BATCH_SIZES = [(10, 20, 2), (5, 25, 5), (2, 10, 1)]  # (start, end, step) for periods
KERNELS = ['auto', 'scalar', 'avx2', 'avx512']

@contextmanager
def measure_time_and_memory():
    """Context manager to measure execution time and memory usage."""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.perf_counter()
    
    yield
    
    end_time = time.perf_counter()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'time': end_time - start_time,
        'memory_delta': mem_after - mem_before
    }

def generate_market_data(size: int) -> np.ndarray:
    """Generate realistic market price data."""
    # Start with a base price
    base_price = 100.0
    
    # Generate returns with some autocorrelation (momentum)
    returns = np.random.normal(0.0001, 0.02, size)
    for i in range(1, len(returns)):
        returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
    
    # Convert to prices
    prices = base_price * np.exp(np.cumsum(returns))
    return prices.astype(np.float64)

def benchmark_single_calculation(data_sizes: List[int], kernels: List[str]) -> Dict:
    """Benchmark single ALMA calculations."""
    results = {}
    
    # Default ALMA parameters
    period = 9
    offset = 0.85
    sigma = 6.0
    
    for kernel in kernels:
        kernel_results = {}
        
        for size in data_sizes:
            data = generate_market_data(size)
            times = []
            
            # Warmup
            for _ in range(WARMUP_RUNS):
                _ = alma(data, period, offset, sigma, kernel=kernel)
            
            # Benchmark
            for _ in range(BENCHMARK_RUNS):
                start = time.perf_counter()
                result = alma(data, period, offset, sigma, kernel=kernel)
                end = time.perf_counter()
                times.append(end - start)
            
            kernel_results[size] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'throughput': size / np.mean(times) / 1e6  # Million elements/second
            }
        
        results[kernel] = kernel_results
    
    return results

def benchmark_batch_calculation(data_size: int, batch_configs: List[Tuple], kernels: List[str]) -> Dict:
    """Benchmark batch ALMA calculations."""
    results = {}
    data = generate_market_data(data_size)
    
    for kernel in kernels:
        kernel_results = {}
        
        for period_range in batch_configs:
            config_key = f"periods_{period_range[0]}-{period_range[1]}_step_{period_range[2]}"
            times = []
            
            # Calculate number of combinations
            periods = list(range(period_range[0], period_range[1] + 1, period_range[2]))
            num_combos = len(periods)
            
            # Warmup
            for _ in range(WARMUP_RUNS):
                _ = alma_batch(data, period_range, (0.85, 0.85, 0.0), (6.0, 6.0, 0.0), kernel=kernel)
            
            # Benchmark
            for _ in range(BENCHMARK_RUNS):
                start = time.perf_counter()
                result = alma_batch(data, period_range, (0.85, 0.85, 0.0), (6.0, 6.0, 0.0), kernel=kernel)
                end = time.perf_counter()
                times.append(end - start)
            
            kernel_results[config_key] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'num_combinations': num_combos,
                'time_per_combo': np.mean(times) / num_combos,
                'throughput': (data_size * num_combos) / np.mean(times) / 1e6  # Million elements/second
            }
        
        results[kernel] = kernel_results
    
    return results

def benchmark_streaming(data_size: int) -> Dict:
    """Benchmark streaming ALMA calculations."""
    data = generate_market_data(data_size)
    
    # Create stream
    stream = AlmaStream(period=9, offset=0.85, sigma=6.0)
    
    times = []
    
    # Warmup
    for _ in range(min(100, data_size)):
        stream.update(data[_])
    
    # Reset stream
    stream = AlmaStream(period=9, offset=0.85, sigma=6.0)
    
    # Benchmark
    start = time.perf_counter()
    for value in data:
        result = stream.update(value)
    end = time.perf_counter()
    
    total_time = end - start
    
    return {
        'total_time': total_time,
        'time_per_update': total_time / data_size,
        'updates_per_second': data_size / total_time
    }

def benchmark_memory_efficiency(data_sizes: List[int]) -> Dict:
    """Benchmark memory usage and zero-copy efficiency."""
    results = {}
    
    for size in data_sizes:
        data = generate_market_data(size)
        
        # Measure memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Single calculation
        result = alma(data, 9, 0.85, 6.0)
        
        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Expected memory usage (roughly 2x data size for input + output)
        expected_mem = (size * 8 * 2) / 1024 / 1024  # MB
        actual_mem = mem_after - mem_before
        
        results[size] = {
            'expected_memory_mb': expected_mem,
            'actual_memory_mb': actual_mem,
            'efficiency_ratio': expected_mem / actual_mem if actual_mem > 0 else float('inf')
        }
    
    return results

def plot_results(single_results: Dict, filename: str = 'alma_benchmark_results.png'):
    """Create visualization of benchmark results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Throughput vs Data Size
    for kernel in KERNELS:
        if kernel in single_results:
            sizes = sorted(single_results[kernel].keys())
            throughputs = [single_results[kernel][s]['throughput'] for s in sizes]
            ax1.plot(sizes, throughputs, marker='o', label=kernel)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Data Size')
    ax1.set_ylabel('Throughput (Million elements/second)')
    ax1.set_title('ALMA Throughput by Data Size and Kernel')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Execution Time vs Data Size
    for kernel in KERNELS:
        if kernel in single_results:
            sizes = sorted(single_results[kernel].keys())
            times = [single_results[kernel][s]['mean_time'] * 1000 for s in sizes]  # Convert to ms
            ax2.plot(sizes, times, marker='o', label=kernel)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Data Size')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('ALMA Execution Time by Data Size and Kernel')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time per Element
    for kernel in KERNELS:
        if kernel in single_results:
            sizes = sorted(single_results[kernel].keys())
            time_per_elem = [single_results[kernel][s]['mean_time'] / s * 1e9 for s in sizes]  # nanoseconds
            ax3.plot(sizes, time_per_elem, marker='o', label=kernel)
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Data Size')
    ax3.set_ylabel('Time per Element (ns)')
    ax3.set_title('ALMA Processing Time per Element')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Speedup vs Scalar
    if 'scalar' in single_results and 'auto' in single_results:
        sizes = sorted(single_results['scalar'].keys())
        speedup = [single_results['scalar'][s]['mean_time'] / single_results['auto'][s]['mean_time'] for s in sizes]
        ax4.plot(sizes, speedup, marker='o', color='green', linewidth=2)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        ax4.set_xscale('log')
        ax4.set_xlabel('Data Size')
        ax4.set_ylabel('Speedup vs Scalar')
        ax4.set_title('ALMA Auto-kernel Speedup over Scalar')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def run_comprehensive_benchmark():
    """Run the complete benchmark suite."""
    print("=" * 80)
    print("ALMA Python Bindings - Comprehensive Performance Benchmark")
    print("=" * 80)
    print()
    
    # System info
    print("System Information:")
    print(f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} threads)")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Python: {os.sys.version.split()[0]}")
    print()
    
    results = {}
    
    # 1. Single calculation benchmarks
    print("Running single calculation benchmarks...")
    single_results = benchmark_single_calculation(DATA_SIZES, KERNELS)
    results['single_calculation'] = single_results
    
    # Print summary
    print("\nSingle Calculation Results (1M elements):")
    print("-" * 60)
    print(f"{'Kernel':<10} {'Time (ms)':<12} {'Throughput (M elem/s)':<20}")
    print("-" * 60)
    for kernel in KERNELS:
        if kernel in single_results and 1_000_000 in single_results[kernel]:
            res = single_results[kernel][1_000_000]
            print(f"{kernel:<10} {res['mean_time']*1000:<12.3f} {res['throughput']:<20.1f}")
    
    # 2. Batch calculation benchmarks
    print("\n\nRunning batch calculation benchmarks...")
    batch_results = benchmark_batch_calculation(10_000, BATCH_SIZES, ['auto', 'scalar'])
    results['batch_calculation'] = batch_results
    
    # Print summary
    print("\nBatch Calculation Results (10K elements):")
    print("-" * 80)
    print(f"{'Kernel':<10} {'Config':<25} {'Time (ms)':<12} {'Combos':<8} {'ms/combo':<12}")
    print("-" * 80)
    for kernel in ['auto', 'scalar']:
        if kernel in batch_results:
            for config, res in batch_results[kernel].items():
                print(f"{kernel:<10} {config:<25} {res['mean_time']*1000:<12.3f} "
                      f"{res['num_combinations']:<8} {res['time_per_combo']*1000:<12.3f}")
    
    # 3. Streaming benchmarks
    print("\n\nRunning streaming benchmarks...")
    stream_results = {}
    for size in [1_000, 10_000, 100_000]:
        stream_results[size] = benchmark_streaming(size)
    results['streaming'] = stream_results
    
    # Print summary
    print("\nStreaming Results:")
    print("-" * 60)
    print(f"{'Data Size':<15} {'Total Time (s)':<15} {'Updates/sec':<15}")
    print("-" * 60)
    for size, res in stream_results.items():
        print(f"{size:<15,} {res['total_time']:<15.6f} {res['updates_per_second']:<15,.0f}")
    
    # 4. Memory efficiency benchmarks
    print("\n\nRunning memory efficiency benchmarks...")
    memory_results = benchmark_memory_efficiency([10_000, 100_000, 1_000_000])
    results['memory_efficiency'] = memory_results
    
    # Print summary
    print("\nMemory Efficiency Results:")
    print("-" * 70)
    print(f"{'Data Size':<15} {'Expected (MB)':<15} {'Actual (MB)':<15} {'Efficiency':<15}")
    print("-" * 70)
    for size, res in memory_results.items():
        print(f"{size:<15,} {res['expected_memory_mb']:<15.2f} {res['actual_memory_mb']:<15.2f} "
              f"{res['efficiency_ratio']:<15.2f}")
    
    # 5. Generate plots
    print("\n\nGenerating performance plots...")
    plot_results(single_results)
    
    # 6. Save results to JSON
    with open('alma_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark complete! Results saved to:")
    print("  - alma_benchmark_results.json (detailed data)")
    print("  - alma_benchmark_results.png (visualization)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    if 'auto' in single_results and 1_000_000 in single_results['auto']:
        auto_1m = single_results['auto'][1_000_000]
        print(f"\nBest throughput (1M elements): {auto_1m['throughput']:.1f} million elements/second")
        print(f"Processing time per element: {auto_1m['mean_time'] / 1_000_000 * 1e9:.1f} nanoseconds")
        
        if 'scalar' in single_results and 1_000_000 in single_results['scalar']:
            scalar_1m = single_results['scalar'][1_000_000]
            speedup = scalar_1m['mean_time'] / auto_1m['mean_time']
            print(f"SIMD speedup over scalar: {speedup:.2f}x")

if __name__ == "__main__":
    run_comprehensive_benchmark()