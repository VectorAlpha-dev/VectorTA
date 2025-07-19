use my_project::indicators::moving_averages::srwma::{srwma_with_kernel, SrwmaInput, SrwmaParams};
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
    // Create test data
    let mut data = vec![0.0; 1_000_000];
    let mut seed = 42u64;
    for i in 0..data.len() {
        // Simple linear congruential generator for reproducible random numbers
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let rand_val = (seed as f64) / (u64::MAX as f64) * 2.0 - 1.0;
        data[i] = rand_val;
    }
    
    let params = SrwmaParams { period: Some(14) };
    let input = SrwmaInput::from_slice(&data, params);
    
    // Warmup
    for _ in 0..10 {
        let _ = srwma_with_kernel(&input, Kernel::Auto).unwrap();
    }
    
    // Single run
    let start = Instant::now();
    let result = srwma_with_kernel(&input, Kernel::Scalar).unwrap();
    let duration = start.elapsed();
    println!("Single run time (Scalar): {:.3} ms", duration.as_secs_f64() * 1000.0);
    println!("Result length: {}", result.values.len());
    println!("Number of NaN values: {}", result.values.iter().filter(|x| x.is_nan()).count());
    
    // Benchmark different kernels
    for kernel in [Kernel::Scalar, Kernel::Avx2, Kernel::Avx512, Kernel::Auto] {
        let mut times = Vec::new();
        for _ in 0..100 {
            let start = Instant::now();
            let _ = srwma_with_kernel(&input, kernel).unwrap();
            let duration = start.elapsed();
            times.push(duration.as_secs_f64() * 1000.0);
        }
        
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times[0];
        let max = times[times.len() - 1];
        
        println!("\nKernel {:?}:", kernel);
        println!("  Median: {:.3} ms", median);
        println!("  Mean: {:.3} ms", mean);
        println!("  Min: {:.3} ms", min);
        println!("  Max: {:.3} ms", max);
    }
}