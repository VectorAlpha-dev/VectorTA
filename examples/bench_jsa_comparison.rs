use my_project::indicators::moving_averages::jsa::{jsa_with_kernel, jsa_with_kernel_into, JsaInput, JsaParams};
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
    // Create test data matching WASM benchmark sizes
    let sizes = vec![
        ("10k", 10_000),
        ("100k", 100_000),
        ("1M", 1_000_000),
    ];
    
    println!("JSA Performance Comparison: Rust vs WASM");
    println!("========================================");
    
    for (size_name, size) in sizes {
        // Generate simple test data
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin() * 100.0 + 1000.0).collect();
        let params = JsaParams { period: Some(30) };
        let input = JsaInput::from_slice(&data, params);
        
        // Test 1: Allocating version (what safe WASM API uses internally)
        let mut alloc_times = Vec::new();
        
        // Warmup
        for _ in 0..100 {
            let _ = jsa_with_kernel(&input, Kernel::Auto).unwrap();
        }
        
        // Measure
        for _ in 0..1000 {
            let start = Instant::now();
            let _ = jsa_with_kernel(&input, Kernel::Auto).unwrap();
            alloc_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        alloc_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alloc_median = alloc_times[alloc_times.len() / 2];
        
        // Test 2: Direct write version (what fast WASM API uses)
        let mut output = vec![0.0; data.len()];
        let mut direct_times = Vec::new();
        
        // Warmup
        for _ in 0..100 {
            jsa_with_kernel_into(&input, Kernel::Auto, &mut output).unwrap();
        }
        
        // Measure
        for _ in 0..1000 {
            let start = Instant::now();
            jsa_with_kernel_into(&input, Kernel::Auto, &mut output).unwrap();
            direct_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        direct_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let direct_median = direct_times[direct_times.len() / 2];
        
        println!("\n{} elements:", size_name);
        println!("  Rust (allocating):   {:.3} ms", alloc_median);
        println!("  Rust (direct write): {:.3} ms", direct_median);
        
        // WASM results from our benchmark
        let (wasm_safe, wasm_fast) = match size_name {
            "10k" => (0.026, 0.002),
            "100k" => (0.159, 0.018),
            "1M" => (1.383, 0.180),
            _ => (0.0, 0.0),
        };
        
        println!("  WASM (safe API):     {:.3} ms", wasm_safe);
        println!("  WASM (fast API):     {:.3} ms", wasm_fast);
        
        println!("\n  Overhead Analysis:");
        println!("    WASM safe vs Rust alloc:    {:.1}x slower", wasm_safe / alloc_median);
        println!("    WASM fast vs Rust direct:   {:.1}x slower", wasm_fast / direct_median);
        println!("    Expected overhead: ~2x");
    }
}