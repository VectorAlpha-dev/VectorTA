use my_project::indicators::moving_averages::ma::{ma, MaData};
use std::time::Instant;

fn main() {
    // Generate test data matching Python benchmark
    // Simple deterministic data to avoid rand dependency
    let data: Vec<f64> = (0..1_000_000).map(|i| ((i as f64) * 0.123).sin()).collect();
    
    // Warmup
    println!("Warming up...");
    for _ in 0..10 {
        let _ = ma("sma", MaData::Slice(&data), 14).unwrap();
    }
    
    // Benchmark
    println!("\nBenchmarking ma function with 'sma' type...");
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = ma("sma", MaData::Slice(&data), 14).unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_time = times[times.len() / 2];
    let mean_time = times.iter().sum::<f64>() / times.len() as f64;
    
    println!("Rust Benchmark Results:");
    println!("Median time: {:.2} ms", median_time);
    println!("Mean time: {:.2} ms", mean_time);
    
    // Test different MA types
    println!("\nTesting different MA types...");
    let ma_types = ["sma", "ema", "wma", "alma", "hma"];
    for ma_type in &ma_types {
        let mut times = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let _ = ma(ma_type, MaData::Slice(&data), 14).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        println!("{}: {:.2} ms", ma_type, median);
    }
}