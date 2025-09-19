//! Test program to measure Rust-only performance for comparison

use my_project::indicators::moving_averages::alma::{alma_with_kernel, AlmaInput, AlmaParams};
use my_project::indicators::moving_averages::sma::{sma_with_kernel, SmaInput, SmaParams};
use my_project::utilities::data_loader::read_candles_from_csv;
use my_project::utilities::enums::Kernel;
use std::time::Instant;

fn main() {
    println!("Loading 1M candles data...");
    let candles =
        read_candles_from_csv("src/data/1MillionCandles.csv").expect("Failed to load CSV");

    let data = &candles.close;
    println!("Loaded {} data points\n", data.len());

    // Warmup
    for _ in 0..10 {
        let _ = sma_with_kernel(
            &SmaInput::from_slice(data, SmaParams { period: Some(14) }),
            Kernel::Auto,
        );
        let _ = alma_with_kernel(
            &AlmaInput::from_slice(
                data,
                AlmaParams {
                    period: Some(9),
                    offset: Some(0.85),
                    sigma: Some(6.0),
                },
            ),
            Kernel::Auto,
        );
    }

    // Benchmark SMA
    println!("Benchmarking Rust-only performance...");
    println!("{:=<60}", "=");

    let mut sma_times = Vec::new();
    for _ in 0..50 {
        let start = Instant::now();
        let _ = sma_with_kernel(
            &SmaInput::from_slice(data, SmaParams { period: Some(14) }),
            Kernel::Auto,
        );
        sma_times.push(start.elapsed());
    }

    let sma_avg = sma_times.iter().sum::<std::time::Duration>() / sma_times.len() as u32;
    println!(
        "SMA (Auto kernel): {:.2} ms",
        sma_avg.as_secs_f64() * 1000.0
    );

    // Benchmark ALMA with Auto kernel
    let mut alma_times = Vec::new();
    for _ in 0..50 {
        let start = Instant::now();
        let _ = alma_with_kernel(
            &AlmaInput::from_slice(
                data,
                AlmaParams {
                    period: Some(9),
                    offset: Some(0.85),
                    sigma: Some(6.0),
                },
            ),
            Kernel::Auto,
        );
        alma_times.push(start.elapsed());
    }

    let alma_avg = alma_times.iter().sum::<std::time::Duration>() / alma_times.len() as u32;
    println!(
        "ALMA (Auto kernel): {:.2} ms",
        alma_avg.as_secs_f64() * 1000.0
    );

    // Now test with explicit kernels
    println!("\nTesting ALMA with explicit kernels:");
    println!("{}", "-".repeat(40));

    // Test ALMA with AVX512
    #[cfg(all(feature = "nightly-avx", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            let mut times = Vec::new();
            for _ in 0..50 {
                let start = Instant::now();
                let _ = alma_with_kernel(
                    &AlmaInput::from_slice(
                        data,
                        AlmaParams {
                            period: Some(9),
                            offset: Some(0.85),
                            sigma: Some(6.0),
                        },
                    ),
                    Kernel::Avx512,
                );
                times.push(start.elapsed());
            }
            let avg = times.iter().sum::<std::time::Duration>() / times.len() as u32;
            println!("ALMA (AVX512): {:.2} ms", avg.as_secs_f64() * 1000.0);
        }

        if std::arch::is_x86_feature_detected!("avx2") {
            let mut times = Vec::new();
            for _ in 0..50 {
                let start = Instant::now();
                let _ = alma_with_kernel(
                    &AlmaInput::from_slice(
                        data,
                        AlmaParams {
                            period: Some(9),
                            offset: Some(0.85),
                            sigma: Some(6.0),
                        },
                    ),
                    Kernel::Avx2,
                );
                times.push(start.elapsed());
            }
            let avg = times.iter().sum::<std::time::Duration>() / times.len() as u32;
            println!("ALMA (AVX2): {:.2} ms", avg.as_secs_f64() * 1000.0);
        }
    }

    // Scalar
    let mut times = Vec::new();
    for _ in 0..50 {
        let start = Instant::now();
        let _ = alma_with_kernel(
            &AlmaInput::from_slice(
                data,
                AlmaParams {
                    period: Some(9),
                    offset: Some(0.85),
                    sigma: Some(6.0),
                },
            ),
            Kernel::Scalar,
        );
        times.push(start.elapsed());
    }
    let avg = times.iter().sum::<std::time::Duration>() / times.len() as u32;
    println!("ALMA (Scalar): {:.2} ms", avg.as_secs_f64() * 1000.0);

    println!("\nNOTE: Python bindings use Kernel::Auto by default,");
    println!("which should select the fastest available kernel.");
}
