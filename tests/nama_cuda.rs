// Integration tests for CUDA NAMA kernels

use my_project::indicators::moving_averages::nama::{
    nama_batch_with_kernel, nama_with_kernel, NamaBatchRange, NamaInput, NamaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaNama;

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
fn nama_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[nama_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 12..series_len {
        let t = i as f64;
        data[i] = (t * 0.0031).sin() + 0.00037 * t;
    }

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let data_quant: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();

    let sweep = NamaBatchRange {
        period: (10, 64, 7),
    };

    let cpu = nama_batch_with_kernel(&data_quant, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaNama::new(0)?;
    let handle = cuda.nama_batch_dev(&data_f32, &sweep)?;

    assert_eq!(handle.rows, cpu.rows);
    assert_eq!(handle.cols, cpu.cols);

    let mut gpu_host = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_host)?;

    let tol = 1e-5;
    for idx in 0..gpu_host.len() {
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
fn nama_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[nama_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in (series + 8)..series_len {
            let time = t as f64;
            let base = (time * 0.0024 + series as f64 * 0.11).sin();
            let drift = 0.00029 * time + 0.013 * series as f64;
            data_tm[t * num_series + series] = base + drift;
        }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let data_tm_quant: Vec<f64> = data_tm_f32.iter().map(|&v| v as f64).collect();

    let params = NamaParams { period: Some(32) };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_data = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_data[t] = data_tm_quant[t * num_series + series];
        }
        let input = NamaInput::from_slice(&series_data, params);
        let out = nama_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = out.values[t];
        }
    }

    let cuda = CudaNama::new(0)?;
    let handle = cuda.nama_many_series_one_param_time_major_dev(
        &data_tm_f32,
        num_series,
        series_len,
        &params,
    )?;

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-5;
    for idx in 0..gpu_tm.len() {
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
