// Integration tests for CUDA HWMA kernels

use my_project::indicators::moving_averages::hwma::{
    hwma_batch_with_kernel, HwmaBatchRange, HwmaBuilder, HwmaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, moving_averages::CudaHwma};

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
fn hwma_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[hwma_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 16..series_len {
        let x = i as f64;
        data[i] = (x * 0.00091).sin() + 0.00037 * x;
    }

    let sweep = HwmaBatchRange {
        na: (0.10, 0.40, 0.10),
        nb: (0.05, 0.25, 0.05),
        nc: (0.05, 0.20, 0.05),
    };

    let cpu = hwma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaHwma::new(0).expect("CudaHwma::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .hwma_batch_dev(&data_f32, &sweep)
        .expect("cuda hwma_batch_dev");

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);

    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda hwma batch result");

    let tol = 7.5e-4f64;
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
fn hwma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[hwma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in j..series_len {
            let idx = t * num_series + j;
            let x = (t as f64) * 0.0017 + (j as f64) * 0.13;
            data_tm[idx] = (x).cos() + 0.00042 * (t as f64);
        }
    }

    let params = HwmaParams {
        na: Some(0.18),
        nb: Some(0.12),
        nc: Some(0.08),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = HwmaBuilder::new()
            .na(params.na.unwrap())
            .nb(params.nb.unwrap())
            .nc(params.nc.unwrap())
            .apply_slice(&series)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    let cuda = CudaHwma::new(0).expect("CudaHwma::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let gpu = cuda
        .hwma_multi_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("cuda hwma_multi_series_one_param_time_major_dev");

    assert_eq!(gpu.rows, series_len);
    assert_eq!(gpu.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu.len()];
    gpu.buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda hwma many-series result");

    let tol = 7.5e-4f64;
    for idx in 0..(num_series * series_len) {
        let a = cpu_tm[idx];
        let b = gpu_tm[idx] as f64;
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
