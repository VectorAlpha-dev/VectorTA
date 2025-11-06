// Integration tests for CUDA SQWMA kernels

use my_project::indicators::moving_averages::sqwma::{
    sqwma_batch_with_kernel, SqwmaBatchRange, SqwmaBuilder,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaSqwma;

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
fn sqwma_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sqwma_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let mut prices = vec![f64::NAN; series_len];
    for i in 8..series_len {
        let x = i as f64;
        prices[i] = (x * 0.0015).sin() + 0.0002 * x;
    }

    let sweep = SqwmaBatchRange { period: (5, 35, 3) };

    // CPU baseline using scalar batch kernel
    let cpu = sqwma_batch_with_kernel(&prices, &sweep, Kernel::ScalarBatch)?;

    // GPU execution
    let cuda = CudaSqwma::new(0).expect("CudaSqwma::new");
    let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .sqwma_batch_dev(&prices_f32, &sweep)
        .expect("cuda sqwma_batch_dev");

    assert_eq!(gpu_handle.rows, cpu.rows);
    assert_eq!(gpu_handle.cols, cpu.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy sqwma batch gpu result");

    let tol = 1e-5;
    for idx in 0..gpu_host.len() {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    // Spot-check builder output for a single period
    let row_idx = cpu
        .combos
        .iter()
        .position(|p| p.period.unwrap() == 14)
        .unwrap();
    let cpu_single = SqwmaBuilder::new().period(14).apply_slice(&prices)?.values;
    let gpu_row = &gpu_host[row_idx * series_len..(row_idx + 1) * series_len];
    for (a, b) in cpu_single.iter().zip(gpu_row.iter()) {
        assert!(approx_eq(*a, *b as f64, tol));
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn sqwma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sqwma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 1024usize;
    let mut prices_tm = vec![f64::NAN; num_series * series_len];
    for series_idx in 0..num_series {
        for t in series_idx..series_len {
            let x = (t as f64) + (series_idx as f64) * 0.25;
            let value = (x * 0.002).cos() + 0.0003 * x;
            prices_tm[t * num_series + series_idx] = value;
        }
    }

    let period = 16usize;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series_idx in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = prices_tm[t * num_series + series_idx];
        }
        let cpu = SqwmaBuilder::new()
            .period(period)
            .apply_slice(&series)?
            .values;
        for t in 0..series_len {
            cpu_tm[t * num_series + series_idx] = cpu[t];
        }
    }

    // GPU execution
    let cuda = CudaSqwma::new(0).expect("CudaSqwma::new");
    let prices_f32: Vec<f32> = prices_tm.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .sqwma_many_series_one_param_time_major_dev(&prices_f32, num_series, series_len, period)
        .expect("cuda sqwma_many_series_one_param_time_major_dev");

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy sqwma many-series gpu result");

    let tol = 1e-5;
    for idx in 0..cpu_tm.len() {
        let cpu_val = cpu_tm[idx];
        let gpu_val = gpu_tm[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}
