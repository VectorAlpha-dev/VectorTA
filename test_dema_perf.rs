use my_project::indicators::moving_averages::dema::{dema_with_kernel, DemaInput, DemaParams};
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
    // Create test data
    let data: Vec<f64> = (0..1_000_000).map(|_| rand::random::<f64>()).collect();
    
    let params = DemaParams {
        period: Some(30),
    };
    let input = DemaInput::from_slice(&data, params);
    
    // Warmup
    for _ in 0..10 {
        let _ = dema_with_kernel(&input, Kernel::Auto);
    }
    
    // Benchmark
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = dema_with_kernel(&input, Kernel::Auto);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times[0];
    let max = times[times.len() - 1];
    
    println!("DEMA Rust Performance (1M points):");
    println!("  Median: {:.2} ms", median);
    println!("  Mean: {:.2} ms", mean);
    println!("  Min: {:.2} ms", min);
    println!("  Max: {:.2} ms", max);
}