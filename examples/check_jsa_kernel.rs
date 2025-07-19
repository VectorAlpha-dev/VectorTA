use my_project::indicators::moving_averages::jsa::{jsa_with_kernel, JsaInput, JsaParams};
use my_project::utilities::enums::Kernel;
use my_project::utilities::helpers::detect_best_kernel;
use std::time::Instant;

fn main() {
    println!("JSA Kernel Performance Check");
    println!("============================");
    
    // Check which kernel is selected
    let best_kernel = detect_best_kernel();
    println!("\nBest kernel detected: {:?}", best_kernel);
    
    // Test data
    let size = 1_000_000;
    let data: Vec<f64> = (0..size).map(|i| (i as f64).sin() * 100.0 + 1000.0).collect();
    let params = JsaParams { period: Some(30) };
    let input = JsaInput::from_slice(&data, params);
    
    // Test each kernel
    let kernels = vec![
        Kernel::Scalar,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx2,
        #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
        Kernel::Avx512,
        Kernel::Auto,
    ];
    
    println!("\nPerformance by kernel (1M elements):");
    println!("------------------------------------");
    
    for kernel in kernels {
        // Warmup
        for _ in 0..10 {
            let _ = jsa_with_kernel(&input, kernel).unwrap();
        }
        
        // Measure
        let mut times = Vec::new();
        for _ in 0..100 {
            let start = Instant::now();
            let _ = jsa_with_kernel(&input, kernel).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        
        println!("{:?}: {:.3} ms", kernel, median);
    }
    
    println!("\nConclusion:");
    println!("- If AVX kernels show same time as Scalar, they're stubs");
    println!("- WASM always uses Scalar (no SIMD support)");
    println!("- This explains the performance gap");
}