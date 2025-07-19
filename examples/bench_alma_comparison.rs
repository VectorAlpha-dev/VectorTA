use my_project::indicators::moving_averages::alma::{alma_with_kernel, AlmaInput, AlmaParams};
use my_project::indicators::moving_averages::jsa::{jsa_with_kernel_into, JsaInput, JsaParams};
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
    // Load CSV data like Python does
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
    
    println!("Loaded {} candles from CSV\n", closes.len());
    let data = closes;
    
    // Benchmark ALMA (allocating)
    println!("ALMA benchmark (allocating):");
    let alma_params = AlmaParams {
        period: Some(9),
        offset: Some(0.85),
        sigma: Some(6.0),
    };
    let alma_input = AlmaInput::from_slice(&data, alma_params);
    
    let mut alma_times = Vec::new();
    for _ in 0..10 {
        let _ = alma_with_kernel(&alma_input, Kernel::Auto).unwrap();
    }
    
    for _ in 0..100 {
        let start = Instant::now();
        let _ = alma_with_kernel(&alma_input, Kernel::Auto).unwrap();
        alma_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    
    alma_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let alma_median = alma_times[alma_times.len() / 2];
    
    // Benchmark JSA (direct write)
    println!("\nJSA benchmark (direct write):");
    let jsa_params = JsaParams { period: Some(30) };
    let jsa_input = JsaInput::from_slice(&data, jsa_params);
    let mut jsa_output = vec![0.0; data.len()];
    
    let mut jsa_times = Vec::new();
    for _ in 0..10 {
        jsa_with_kernel_into(&jsa_input, Kernel::Auto, &mut jsa_output).unwrap();
    }
    
    for _ in 0..100 {
        let start = Instant::now();
        jsa_with_kernel_into(&jsa_input, Kernel::Auto, &mut jsa_output).unwrap();
        jsa_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    
    jsa_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let jsa_median = jsa_times[jsa_times.len() / 2];
    
    println!("\n{}", "=".repeat(60));
    println!("Rust Baseline Performance:");
    println!("  ALMA (allocating): {:.3} ms", alma_median);
    println!("  JSA (direct write): {:.3} ms", jsa_median);
    
    println!("\nExpected Python binding times with <10% overhead:");
    println!("  ALMA: {:.3} - {:.3} ms", alma_median, alma_median * 1.1);
    println!("  JSA: {:.3} - {:.3} ms", jsa_median, jsa_median * 1.1);
    
    println!("\nActual Python binding times (from previous runs):");
    println!("  ALMA: ~0.961 ms");
    println!("  JSA: ~0.590 ms");
    
    // Calculate actual overhead
    let alma_overhead = ((0.961 - alma_median) / alma_median) * 100.0;
    let jsa_overhead = ((0.590 - jsa_median) / jsa_median) * 100.0;
    
    println!("\nActual overhead:");
    println!("  ALMA: {:.1}%", alma_overhead);
    println!("  JSA: {:.1}%", jsa_overhead);
}