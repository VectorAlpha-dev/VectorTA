use cross_library_benchmark::unified_benchmark::{UnifiedBenchmarkRunner, LibraryType};
use cross_library_benchmark::benchmark_methodology::ComparisonMode;
use cross_library_benchmark::{rust_ffi, tulip};
use cross_library_benchmark::utils::CandleData;
use my_project::indicators::moving_averages::sma;
use std::path::Path;

fn main() {
    println!("🚀 Unified Three-Tier Benchmark System Demo\n");

    // Create the unified runner
    let mut runner = UnifiedBenchmarkRunner::new();

    // Step 1: Profile FFI overhead ONCE
    println!("Step 1: Profiling FFI overhead...");
    runner.profile_ffi_overhead(10000, 1000);

    // Load test data
    let data_path = Path::new("../../src/data/10kCandles.csv");
    let data = if data_path.exists() {
        CandleData::from_csv(data_path).expect("Failed to load data")
    } else {
        // Create synthetic data if file doesn't exist
        println!("Using synthetic data for demo...");
        CandleData {
            timestamps: vec![0; 10000],
            open: vec![100.0; 10000],
            high: vec![105.0; 10000],
            low: vec![95.0; 10000],
            close: (0..10000).map(|i| 100.0 + (i as f64 * 0.01)).collect(),
            volume: vec![1000.0; 10000],
        }
    };

    println!("\nStep 2: Running benchmarks (once per library)...");
    let iterations = 100;

    // Benchmark 1: Rust Native (direct call)
    println!("  • Benchmarking Rust Native SMA...");
    runner.benchmark(
        "SMA",
        LibraryType::RustNative,
        data.len(),
        iterations,
        || {
            let input = sma::SmaInput::from_slice(&data.close, sma::SmaParams { period: Some(14) });
            let _ = sma::sma(&input);
        },
    );

    // Benchmark 2: Rust FFI (through C interface)
    println!("  • Benchmarking Rust FFI SMA...");
    let mut rust_output = vec![0.0; data.len()];
    runner.benchmark(
        "SMA",
        LibraryType::RustFFI,
        data.len(),
        iterations,
        || {
            unsafe {
                rust_ffi::rust_sma(
                    data.len() as i32,
                    data.close.as_ptr(),
                    14,
                    rust_output.as_mut_ptr(),
                );
            }
        },
    );

    // Benchmark 3: Tulip (C library through FFI)
    println!("  • Benchmarking Tulip SMA...");
    let mut tulip_output = vec![0.0; data.len()];
    runner.benchmark(
        "SMA",
        LibraryType::TulipFFI,
        data.len(),
        iterations,
        || {
            unsafe {
                let inputs = vec![&data.close[..]];
                let options = vec![14.0];
                let mut outputs = vec![&mut tulip_output[..]];
                let _ = tulip::call_indicator(
                    "sma",
                    data.len(),
                    &inputs,
                    &options,
                    &mut outputs,
                );
            }
        },
    );

    // Step 3: Generate three comparison reports from the SAME measurements
    println!("\nStep 3: Generating three-tier analysis from single measurement set...");

    println!("\n{}", "=".repeat(80));
    println!("📊 RAW PERFORMANCE (What Users Experience)");
    println!("{}", "=".repeat(80));
    for result in runner.generate_comparison(ComparisonMode::RawPerformance) {
        println!("{:<15} {:>10.2} µs  {:>10.2} MOPS",
            result.library,
            result.raw_duration.as_secs_f64() * 1_000_000.0,
            result.throughput_mops
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("🧮 ALGORITHM EFFICIENCY (FFI Overhead Removed)");
    println!("{}", "=".repeat(80));
    for result in runner.generate_comparison(ComparisonMode::AlgorithmEfficiency) {
        let duration = result.ffi_compensated_duration.unwrap_or(result.raw_duration);
        println!("{:<15} {:>10.2} µs  {:>10.2} MOPS",
            result.library,
            duration.as_secs_f64() * 1_000_000.0,
            result.throughput_mops
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("⚖️ EQUAL FOOTING (All Through FFI)");
    println!("{}", "=".repeat(80));
    for result in runner.generate_comparison(ComparisonMode::EqualFooting) {
        println!("{:<15} {:>10.2} µs  {:>10.2} MOPS",
            result.library,
            result.raw_duration.as_secs_f64() * 1_000_000.0,
            result.throughput_mops
        );
    }

    // Show statistics
    println!("{}", runner.get_statistics());

    // Generate full report
    println!("\n📄 Full Report:");
    println!("{}", runner.generate_full_report());

    println!("\n✅ Key Insight: We ran each benchmark ONCE but analyzed it THREE ways!");
    println!("   This is much more efficient than running three separate benchmark suites.");
}