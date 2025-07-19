use my_project::indicators::moving_averages::jsa::{jsa_with_kernel_into, JsaInput, JsaParams};
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
    // Load CSV data like WASM benchmark does
    let csv_path = "src/data/1MillionCandles.csv";
    let mut reader = csv::Reader::from_path(csv_path).expect("Failed to open CSV");
    let mut closes = Vec::new();
    
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
    
    // Test with different data sizes to match WASM benchmark
    let sizes = vec![
        ("10k", &data[..10_000]),
        ("100k", &data[..100_000]),
        ("1M", &data[..]),
    ];
    
    println!("\nRust JSA Direct Write Performance:");
    println!("==================================");
    
    for (size_name, data_slice) in sizes {
        let params = JsaParams { period: Some(30) };
        let input = JsaInput::from_slice(data_slice, params);
        let mut output = vec![0.0; data_slice.len()];
        
        // Warmup
        for _ in 0..100 {
            jsa_with_kernel_into(&input, Kernel::Auto, &mut output).unwrap();
        }
        
        // Measure
        let mut times = Vec::new();
        let iterations = 1000;
        
        for _ in 0..iterations {
            let start = Instant::now();
            jsa_with_kernel_into(&input, Kernel::Auto, &mut output).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        let min = times[0];
        let max = times[times.len() - 1];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        
        println!("\nJSA Rust {} (direct write):", size_name);
        println!("  Median: {:.3} ms", median);
        println!("  Mean:   {:.3} ms", mean);
        println!("  Min:    {:.3} ms", min);
        println!("  Max:    {:.3} ms", max);
        println!("  Throughput: {:.1} M elem/s", data_slice.len() as f64 / median / 1000.0);
    }
    
    println!("\n==================================");
    println!("WASM Fast API Performance (from benchmark):");
    println!("  10k:   0.002 ms (5.4 M elem/s)");
    println!("  100k:  0.018 ms (5.5 M elem/s)");
    println!("  1M:    0.180 ms (5.6 M elem/s)");
    println!("\nExpected: WASM should be ~2x slower than Rust");
}