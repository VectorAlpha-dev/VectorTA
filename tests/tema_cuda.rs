// Integration tests for CUDA TEMA kernels

use my_project::indicators::moving_averages::tema::{
    tema_batch_with_kernel, TemaBatchRange, TemaBuilder,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaTema;

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
fn tema_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[tema_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.0015).sin() + 0.0002 * x;
    }

    let sweep = TemaBatchRange { period: (5, 64, 3) };

    let cpu = match tema_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(e)),
    };

    let cuda = CudaTema::new(0).expect("CudaTema::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .tema_batch_dev(&data_f32, &sweep)
        .expect("cuda tema_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda tema batch result to host");

    let tol = 1e-5;
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
fn tema_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[tema_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 3usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in j..series_len {
            let x = (t as f64) + (j as f64) * 0.25;
            data_tm[t * num_series + j] = (x * 0.002).cos() + 0.0003 * x;
        }
    }

    let period = 21;

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = match TemaBuilder::default().period(period).apply_slice(&series) {
            Ok(v) => v,
            Err(e) => return Err(Box::new(e)),
        };
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    let cuda = CudaTema::new(0).expect("CudaTema::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .tema_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, period)
        .expect("cuda tema_many_series_one_param_time_major_dev");

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy many-series result to host");

    let tol = 1e-5;
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
