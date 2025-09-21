use std::time::Instant;
use cross_library_benchmark::{tulip, rust_ffi};
use my_project::indicators::moving_averages::sma;

fn main() {
    println!("Quick Performance Comparison: SMA\n");
    println!("=================================\n");

    // Test different data sizes
    let sizes = vec![1000, 10000, 100000];
    let period = 14;
    let iterations = 100;

    for size in sizes {
        println!("Data size: {} elements", size);
        println!("-----------------------");

        // Generate test data
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin() * 100.0 + 50.0).collect();

        // Benchmark Rust Native (with SIMD if available)
        let input = sma::SmaInput::from_slice(&data, sma::SmaParams { period: Some(period) });
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sma::sma(&input).unwrap();
        }
        let rust_duration = start.elapsed();
        let rust_per_iter = rust_duration / iterations;
        let rust_throughput = (size as f64 / rust_per_iter.as_secs_f64()) / 1_000_000.0;

        println!("  Rust Native:");
        println!("    Time per iteration: {:.2} µs", rust_per_iter.as_micros() as f64 / 1000.0);
        println!("    Throughput: {:.2} MOPS", rust_throughput);

        // Benchmark Tulip FFI
        unsafe {
            let mut output = vec![0.0; size];
            let options = vec![period as f64];
            let inputs = vec![&data[..]];
            let mut outputs = vec![&mut output[..]];

            let start = Instant::now();
            for _ in 0..iterations {
                tulip::call_indicator(
                    "sma",
                    size,
                    &inputs,
                    &options,
                    &mut outputs,
                ).unwrap();
            }
            let tulip_duration = start.elapsed();
            let tulip_per_iter = tulip_duration / iterations;
            let tulip_throughput = (size as f64 / tulip_per_iter.as_secs_f64()) / 1_000_000.0;

            println!("  Tulip FFI:");
            println!("    Time per iteration: {:.2} µs", tulip_per_iter.as_micros() as f64 / 1000.0);
            println!("    Throughput: {:.2} MOPS", tulip_throughput);

            // Calculate speedup
            let speedup = rust_throughput / tulip_throughput;
            if speedup > 1.0 {
                println!("  🚀 Rust is {:.2}x faster!", speedup);
            } else {
                println!("  ⚠️  Tulip is {:.2}x faster", 1.0 / speedup);
            }
        }

        println!();
    }

    // Check if SIMD is being used
    #[cfg(feature = "nightly-avx")]
    {
        println!("✅ SIMD optimizations (nightly-avx) are ENABLED");

        use my_project::utilities::helpers::detect_best_kernel;
        let kernel = detect_best_kernel();
        println!("   Using kernel: {:?}", kernel);
    }

    #[cfg(not(feature = "nightly-avx"))]
    {
        println!("⚠️  SIMD optimizations (nightly-avx) are DISABLED");
        println!("   To enable, run with: --features nightly-avx");
    }
}