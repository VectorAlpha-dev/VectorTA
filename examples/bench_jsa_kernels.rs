use my_project::indicators::moving_averages::jsa::{jsa_with_kernel, JsaInput, JsaParams};
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn benchmark_kernel(data: &[f64], kernel: Kernel, name: &str) {
    let params = JsaParams { period: Some(30) };
    let input = JsaInput::from_slice(data, params);
    
    // Warmup
    for _ in 0..10 {
        let _ = jsa_with_kernel(&input, kernel).unwrap();
    }
    
    // Benchmark
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = jsa_with_kernel(&input, kernel).unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    
    println!("  Kernel '{}': {:.3} ms (min: {:.3}, max: {:.3})", 
             name, median, times[0], times[times.len() - 1]);
}

fn main() {
    // Create test data - same as Python
    let data: Vec<f64> = (0..1_002_240)
        .map(|i| 50000.0 + (i as f64).sin() * 1000.0)
        .collect();
    
    println!("JSA Rust benchmark by kernel:");
    println!("  Data size: {} points", data.len());
    println!();
    
    benchmark_kernel(&data, Kernel::Auto, "auto");
    benchmark_kernel(&data, Kernel::Scalar, "scalar");
    
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        benchmark_kernel(&data, Kernel::Avx2, "avx2");
        benchmark_kernel(&data, Kernel::Avx512, "avx512");
    }
}