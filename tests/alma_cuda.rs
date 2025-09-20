// Integration tests for CUDA ALMA kernels

use my_project::indicators::moving_averages::alma::{
    alma_batch_with_kernel, AlmaBatchRange, AlmaBuilder, AlmaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaAlma;
#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    // This test ensures the file compiles/runs when `cuda` feature is disabled.
    #[cfg(not(feature = "cuda"))]
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn alma_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[alma_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    // Synthetic series with NaNs prefix
    let series_len = 2048usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 3..series_len {
        let x = i as f64;
        data[i] = (x * 0.001).sin() + 0.0001 * x;
    }

    let sweep = AlmaBatchRange {
        period: (9, 32, 1),
        offset: (0.85, 0.85, 0.0),
        sigma: (6.0, 6.0, 0.0),
    };

    // CPU baseline (scalar batch)
    let cpu = match alma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(e)),
    };

    // GPU (device handle, copy back for comparison)
    let cuda = CudaAlma::new(0).expect("CudaAlma::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .alma_batch_dev(&data_f32, &sweep)
        .expect("cuda alma_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda alma batch result to host");

    // fp32 kernel vs fp64 CPU: allow a modest tolerance
    let tol = 1e-5;
    for i in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[i];
        let b = gpu_host[i] as f64;
        assert!(
            approx_eq(a, b, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }

    Ok(())
}

// multi-stream variant removed

#[cfg(feature = "cuda")]
#[test]
fn alma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[alma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 1024usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];

    // Build per-series data with varying NaN prefixes (first_valid at j)
    for j in 0..num_series {
        for t in (j)..series_len {
            let x = (t as f64) + (j as f64) * 0.1;
            data_tm[t * num_series + j] = (x * 0.003).cos() + 0.001 * x;
        }
    }

    let params = AlmaParams {
        period: Some(14),
        offset: Some(0.85),
        sigma: Some(6.0),
    };

    // CPU baseline per series (row-major to time-major)
    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = match AlmaBuilder::default()
            .period(params.period.unwrap())
            .offset(params.offset.unwrap())
            .sigma(params.sigma.unwrap())
            .apply_slice(&series)
        {
            Ok(v) => v,
            Err(e) => return Err(Box::new(e)),
        };
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    // GPU
    let cuda = CudaAlma::new(0).expect("CudaAlma::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .alma_multi_series_one_param_time_major_dev(
            &data_tm_f32,
            num_series,
            series_len,
            &params,
        )
        .expect("cuda alma_multi_series_one_param_time_major_dev");

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy many-series result to host");

    let tol = 1e-5;
    for i in 0..(num_series * series_len) {
        let a = cpu_tm[i];
        let b = gpu_tm[i] as f64;
        assert!(
            approx_eq(a, b, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }
    Ok(())
}
