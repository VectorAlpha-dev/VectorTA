use my_project::indicators::moving_averages::alma::{alma, AlmaInput, AlmaParams};
use std::time::Instant;

fn main() {
    // Generate same size data as Python benchmark
    let data: Vec<f64> = (0..1_000_000)
        .map(|i| (i as f64).sin() * 10.0 + 100.0)  // Deterministic data
        .collect();
    
    let params = AlmaParams { 
        period: Some(9), 
        offset: Some(0.85), 
        sigma: Some(6.0) 
    };
    let input = AlmaInput::from_slice(&data, params);
    
    println!("ALMA Rust Benchmark");
    println!("===================");
    println!("Data size: 1,000,000 elements");
    println!("Period: 9, Offset: 0.85, Sigma: 6.0\n");
    
    // Warmup
    println!("Warming up...");
    for _ in 0..10 {
        let _ = alma(&input).unwrap();
    }
    
    // Benchmark
    println!("Running benchmark (100 iterations)...\n");
    let mut times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = alma(&input).unwrap();
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
    }
    
    // Calculate statistics
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times[0];
    let max = times[times.len() - 1];
    let median = times[times.len() / 2];
    
    println!("Results:");
    println!("--------");
    println!("  Mean:   {:.3} ms", mean);
    println!("  Median: {:.3} ms", median);
    println!("  Min:    {:.3} ms", min);
    println!("  Max:    {:.3} ms", max);
    println!("  Throughput: {:.1} M elements/sec", 1_000_000.0 / (mean / 1000.0) / 1e6);
    
    // Show kernel detection
    use my_project::utilities::helpers::detect_best_kernel;
    let kernel = detect_best_kernel();
    println!("\nKernel used: {:?}", kernel);
}