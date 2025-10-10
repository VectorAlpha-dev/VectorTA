// Integration tests for CUDA VWAP kernels

use my_project::indicators::moving_averages::vwap::{vwap_batch_with_kernel, VwapBatchRange};
use my_project::utilities::enums::Kernel;
use my_project::indicators::moving_averages::vwap::{VwapInput, VwapParams, vwap_with_kernel};

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

#[cfg(feature = "cuda")]
#[test]
fn vwap_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vwap_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 2048usize;
    let mut timestamps = vec![0i64; rows];
    for t in 0..rows { timestamps[t] = 1_600_000_000_000i64 + (t as i64) * 60_000; }

    let mut price_tm = vec![f64::NAN; rows * cols];
    let mut volume_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        for t in (s % 5)..rows {
            let x = (t as f64) + (s as f64) * 0.25;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
            volume_tm[t * cols + s] = (x * 0.001).cos().abs() + 0.6;
        }
    }

    let anchor = "1m".to_string();

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; rows * cols];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = price_tm[t * cols + s];
            v[t] = volume_tm[t * cols + s];
        }
        let params = VwapParams { anchor: Some(anchor.clone()) };
        let input = VwapInput::from_slice(&timestamps, &v, &p, params);
        let out = vwap_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows { cpu_tm[t * cols + s] = out.values[t]; }
    }

    let cuda = CudaVwap::new(0).expect("CudaVwap::new");
    let dev = cuda
        .vwap_many_series_one_param_time_major_dev(
            &timestamps,
            &volume_tm,
            &price_tm,
            cols,
            rows,
            &anchor,
        )
        .expect("vwap many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-4;
    for idx in 0..gpu_tm.len() {
        let a = cpu_tm[idx];
        let b = gpu_tm[idx] as f64;
        assert!(approx_eq(a, b, tol), "mismatch at {}", idx);
    }

    Ok(())
}
