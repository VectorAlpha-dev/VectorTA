// Integration tests for CUDA VWAP kernels

use my_project::indicators::moving_averages::vwap::{vwap_batch_with_kernel, VwapBatchRange};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaVwap;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn vwap_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vwap_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let base_ts = 1_600_000_000_000i64;

    let mut timestamps = Vec::with_capacity(series_len);
    let mut prices = Vec::with_capacity(series_len);
    let mut volumes = Vec::with_capacity(series_len);
    for i in 0..series_len {
        timestamps.push(base_ts + (i as i64) * 60_000); // 1-minute spacing
        let x = i as f64;
        prices.push(100.0 + (x * 0.01).sin() + 0.05 * (x * 0.001).cos());
        let vol = if i % 16 == 0 {
            0.0
        } else {
            1.0 + (x * 0.05).sin().abs()
        };
        volumes.push(vol);
    }

    let sweep = VwapBatchRange {
        anchor: ("1m".to_string(), "3m".to_string(), 1),
    };

    let cpu = vwap_batch_with_kernel(&timestamps, &volumes, &prices, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaVwap::new(0).expect("CudaVwap::new");
    let gpu = cuda
        .vwap_batch_dev(&timestamps, &volumes, &prices, &sweep)
        .expect("cuda vwap_batch_dev");

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda vwap batch result to host");

    let tol = 1e-4f64; // fp32 vs fp64 tolerance
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}
