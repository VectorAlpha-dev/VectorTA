use my_project::indicators::moving_averages::ma::{ma, MaData};
use std::time::Instant;

fn main() {
    // Generate test data
    let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
    
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
    let min_time = times[0];
    let max_time = times[times.len() - 1];
    
    println!("Median time: {:.2} ms", median_time);
    println!("Mean time: {:.2} ms", mean_time);
    println!("Min time: {:.2} ms", min_time);
    println!("Max time: {:.2} ms", max_time);
    
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