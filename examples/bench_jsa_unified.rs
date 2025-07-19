use my_project::indicators::moving_averages::jsa::{jsa_with_kernel, jsa_with_kernel_into, JsaInput, JsaParams};
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
    // Load CSV data to match Python benchmark exactly
    let csv_path = "src/data/1MillionCandles.csv";
    let mut reader = csv::Reader::from_path(csv_path).expect("Failed to open CSV");
    let mut closes = Vec::new();
    
    // Skip header and read close prices
    for result in reader.records() {
        if let Ok(record) = result {
            if let Some(close_str) = record.get(4) {
                if let Ok(close) = close_str.parse::<f64>() {
                    closes.push(close);
                }
            }
        }
    }
    
    println!("Loaded {} candles from CSV", closes.len());
    let data = closes;
    
    let params = JsaParams { period: Some(30) };
    let input = JsaInput::from_slice(&data, params);
    
    // Test 1: Allocating version (what Python was using before)
    println!("\n1. Allocating version (jsa_with_kernel):");
    let mut times_alloc = Vec::new();
    
    // Warmup
    for _ in 0..10 {
        let _ = jsa_with_kernel(&input, Kernel::Auto).unwrap();
    }
    
    // Benchmark
    for _ in 0..100 {
        let start = Instant::now();
        let _ = jsa_with_kernel(&input, Kernel::Auto).unwrap();
        times_alloc.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    
    times_alloc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_alloc = times_alloc[times_alloc.len() / 2];
    
    // Test 2: Direct write version (what Python uses now)  
    println!("\n2. Direct write version (jsa_with_kernel_into):");
    let mut times_direct = Vec::new();
    let mut output = vec![0.0; data.len()];
    
    // Warmup
    for _ in 0..10 {
        jsa_with_kernel_into(&input, Kernel::Auto, &mut output).unwrap();
    }
    
    // Benchmark
    for _ in 0..100 {
        let start = Instant::now();
        jsa_with_kernel_into(&input, Kernel::Auto, &mut output).unwrap();
        times_direct.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    
    times_direct.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_direct = times_direct[times_direct.len() / 2];
    
    println!("\nResults:");
    println!("  Allocating version: {:.3} ms", median_alloc);
    println!("  Direct write version: {:.3} ms", median_direct);
    println!("  Difference: {:.3} ms ({:.1}%)", 
             median_alloc - median_direct, 
             ((median_alloc - median_direct) / median_direct) * 100.0);
    
    println!("\nThis is the baseline Rust performance.");
    println!("Python binding should have <10% overhead over the direct write version.");
    println!("Expected Python time: {:.3} - {:.3} ms", median_direct, median_direct * 1.1);
}