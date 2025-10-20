// Integration tests for CUDA decycler kernels

use my_project::indicators::decycler::{
    decycler_batch_with_kernel, decycler_with_kernel, DecyclerBatchRange, DecyclerInput,
    DecyclerParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaDecycler;

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
fn decycler_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[decycler_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00019 * x.cos();
    }

    let sweep = DecyclerBatchRange {
        hp_period: (6, 64, 7),
        k: (0.2, 0.9, 0.15),
    };
    let cpu = decycler_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaDecycler::new(0).expect("CudaDecycler::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .decycler_batch_dev(&data_f32, &sweep)
        .expect("cuda decycler batch dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy decycler cuda batch result to host");

    let tol = 2e-5;
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
fn decycler_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[decycler_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for s in 0..num_series {
        for t in (s * 2)..series_len {
            let base = t as f64 + (s as f64) * 0.41;
            data_tm[t * num_series + s] = (base * 0.0021).cos() + 0.00037 * base.sin();
        }
    }

    let params = DecyclerParams {
        hp_period: Some(48),
        k: Some(0.707),
    };

    // CPU reference: per series
    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for s in 0..num_series {
        let mut one = vec![f64::NAN; series_len];
        for t in 0..series_len {
            one[t] = data_tm[t * num_series + s];
        }
        let input = DecyclerInput::from_slice(&one, params.clone());
        let out = decycler_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..series_len {
            cpu_tm[t * num_series + s] = out[t];
        }
    }

    let cuda = CudaDecycler::new(0).expect("CudaDecycler::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let handle = cuda
        .decycler_many_series_one_param_time_major_dev(
            &data_tm_f32,
            num_series,
            series_len,
            &params,
        )
        .expect("cuda decycler many-series dev");

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy decycler cuda many-series result to host");

    let tol = 2e-5;
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
