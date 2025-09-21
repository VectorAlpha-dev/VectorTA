// Integration tests for CUDA SAMA kernels

use my_project::indicators::moving_averages::sama::{
    sama_batch_with_kernel, sama_with_kernel, SamaBatchRange, SamaInput, SamaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaSama;

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
fn sama_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sama_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 7..series_len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00037 * x.cos();
    }

    let sweep = SamaBatchRange {
        length: (32, 96, 16),
        maj_length: (10, 22, 6),
        min_length: (4, 12, 4),
    };

    let cpu = sama_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaSama::new(0).expect("CudaSama::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .sama_batch_dev(&data_f32, &sweep)
        .expect("sama cuda batch dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy sama cuda batch result to host");

    let tol = 5e-5;
    for idx in 0..(cpu.rows * cpu.cols) {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "batch mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn sama_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sama_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in series..series_len {
            let base = (t as f64) + (series as f64) * 0.43;
            data_tm[t * num_series + series] = (base * 0.0021).cos() + 0.00031 * base;
        }
    }

    let params = SamaParams {
        length: Some(64),
        maj_length: Some(18),
        min_length: Some(8),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut single = vec![f64::NAN; series_len];
        for t in 0..series_len {
            single[t] = data_tm[t * num_series + series];
        }
        let input = SamaInput::from_slice(&single, params.clone());
        let cpu_vals = sama_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = cpu_vals.values[t];
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaSama::new(0).expect("CudaSama::new");
    let handle = cuda
        .sama_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("sama cuda many-series dev");

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy sama cuda many-series result to host");

    let tol = 5e-5;
    for idx in 0..gpu_tm.len() {
        let cpu_val = cpu_tm[idx];
        let gpu_val = gpu_tm[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "many-series mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}
