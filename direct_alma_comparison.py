#!/usr/bin/env python3
"""
Direct comparison of ALMA performance between Python bindings and native Rust.
This script creates a simple Rust benchmark to compare with Python.
"""

import numpy as np
import time
import subprocess
import os
from my_project import alma

def benchmark_python():
    """Benchmark Python bindings."""
    # Create test data
    np.random.seed(42)
    data = np.random.randn(1_000_000).astype(np.float64) * 10 + 100
    
    # Parameters
    period = 9
    offset = 0.85
    sigma = 6.0
    
    # Warmup
    for _ in range(10):
        _ = alma(data, period, offset, sigma)
    
    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = alma(data, period, offset, sigma)
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = np.mean(times) * 1000  # Convert to ms
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    
    print("Python Binding Performance:")
    print(f"  Mean: {mean_time:.3f} ms")
    print(f"  Min:  {min_time:.3f} ms")  
    print(f"  Max:  {max_time:.3f} ms")
    print(f"  Throughput: {1_000_000 / (mean_time / 1000) / 1e6:.1f} M elements/sec")
    
    return mean_time

def create_rust_benchmark():
    """Create and run a simple Rust benchmark."""
    rust_code = '''
use my_project::indicators::moving_averages::alma::{alma, AlmaInput, AlmaParams};
use std::time::Instant;

fn main() {
    // Generate same data as Python
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let data: Vec<f64> = (0..1_000_000)
        .map(|_| rng.gen::<f64>() * 10.0 + 100.0)
        .collect();
    
    let params = AlmaParams { 
        period: Some(9), 
        offset: Some(0.85), 
        sigma: Some(6.0) 
    };
    let input = AlmaInput::from_slice(&data, params);
    
    // Warmup
    for _ in 0..10 {
        let _ = alma(&input).unwrap();
    }
    
    // Benchmark
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = alma(&input).unwrap();
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
    }
    
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("Rust Native Performance:");
    println!("  Mean: {:.3} ms", mean);
    println!("  Min:  {:.3} ms", min);
    println!("  Max:  {:.3} ms", max);
    println!("  Throughput: {:.1} M elements/sec", 1_000_000.0 / (mean / 1000.0) / 1e6);
}
'''
    
    # Write the Rust code
    with open('/tmp/alma_bench.rs', 'w') as f:
        f.write(rust_code)
    
    print("\nCreating and running Rust benchmark...")
    
    # Build the Rust benchmark
    try:
        # First build in release mode
        subprocess.run([
            'cargo', 'build', '--release'
        ], cwd='/mnt/c/Rust Projects/my_project', check=True)
        
        # Compile the benchmark
        subprocess.run([
            'rustc',
            '/tmp/alma_bench.rs',
            '--edition', '2021',
            '-O',  # Optimize
            '-C', 'target-cpu=native',  # Use native CPU features
            '-L', '/mnt/c/Rust Projects/my_project/target/release/deps',
            '--extern', 'my_project=/mnt/c/Rust Projects/my_project/target/release/libmy_project.rlib',
            '--extern', 'rand',
            '-o', '/tmp/alma_bench'
        ], check=True)
        
        # Run the benchmark
        result = subprocess.run(['/tmp/alma_bench'], capture_output=True, text=True)
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Rust benchmark: {e}")
        print("Trying alternative approach...")

def main():
    print("ALMA Performance Investigation")
    print("=" * 60)
    print("\nTesting with 1,000,000 elements, period=9, offset=0.85, sigma=6.0")
    print()
    
    # Run Python benchmark
    python_time = benchmark_python()
    
    # Try to run Rust benchmark
    create_rust_benchmark()
    
    print("\n" + "=" * 60)
    print("\nAnalysis:")
    print("---------")
    print("If Python bindings show ~1.2ms performance, this suggests:")
    print("1. The Python module is compiled with full optimizations")
    print("2. Zero-copy operations are working efficiently")
    print("3. GIL release allows full CPU utilization")
    print("\nPossible reasons for discrepancy with your Rust benchmark:")
    print("1. Different compiler optimization flags")
    print("2. Different CPU frequency scaling during benchmarks")
    print("3. Different memory allocation patterns")
    print("4. Benchmark methodology differences (warmup, measurement)")

if __name__ == "__main__":
    main()