use my_project::indicators::moving_averages::ma::{ma_with_kernel, MaData};
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
    // Generate test data matching Python benchmark
    let data: Vec<f64> = (0..1_000_000).map(|i| ((i as f64) * 0.123).sin()).collect();
    
    // Test different kernels
    println!("Testing ma_with_kernel with different kernels...");
    println!("{}", "=".repeat(60));
    
    let kernels = [
        ("Auto", Kernel::Auto),
        ("Scalar", Kernel::Scalar),
        ("AVX2", Kernel::Avx2),
        ("AVX512", Kernel::Avx512),
    ];
    
    // Warmup
    for _ in 0..10 {
        let _ = ma_with_kernel("sma", MaData::Slice(&data), 14, Kernel::Auto).unwrap();
    }
    
    // Benchmark each kernel
    for (name, kernel) in &kernels {
        let mut times = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let _ = ma_with_kernel("sma", MaData::Slice(&data), 14, *kernel).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_time = times[times.len() / 2];
        println!("{:7} kernel: {:.2} ms", name, median_time);
    }
    
    // Test different MA types with Auto kernel
    println!("\nTesting different MA types with Auto kernel...");
    println!("{}", "=".repeat(60));
    
    let ma_types = ["sma", "ema", "wma", "alma", "hma"];
    for ma_type in &ma_types {
        let mut times = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let _ = ma_with_kernel(ma_type, MaData::Slice(&data), 14, Kernel::Auto).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        println!("{}: {:.2} ms", ma_type, median);
    }
}