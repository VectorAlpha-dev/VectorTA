use my_project::indicators::moving_averages::jsa::{jsa, JsaInput, JsaParams};
use std::time::Instant;

fn main() {
    // Create test data - 1 million points
    let data: Vec<f64> = (0..1_002_240)
        .map(|i| 50000.0 + (i as f64).sin() * 1000.0)
        .collect();
    
    let params = JsaParams { period: Some(30) };
    let input = JsaInput::from_slice(&data, params);
    
    // Warmup
    for _ in 0..10 {
        let _ = jsa(&input).unwrap();
    }
    
    // Benchmark
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = jsa(&input).unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    
    println!("JSA Rust benchmark:");
    println!("  Data size: {} points", data.len());
    println!("  Median time: {:.3} ms", median);
    println!("  Min time: {:.3} ms", times[0]);
    println!("  Max time: {:.3} ms", times[times.len() - 1]);
}