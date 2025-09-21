// Integration tests for CUDA UMA kernels

use my_project::indicators::moving_averages::uma::{
    uma_batch_with_kernel, UmaBatchRange, UmaBuilder, UmaParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaUma;

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
fn uma_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[uma_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let mut prices = vec![f64::NAN; series_len];
    let mut volumes = vec![0.0f64; series_len];
    let first_valid = 16usize;
    for i in first_valid..series_len {
        let x = i as f64;
        prices[i] = (x * 0.002).sin() + 0.0005 * x;
        volumes[i] = 1000.0 + (x * 0.0007).cos() * 250.0;
    }

    let sweep = UmaBatchRange {
        accelerator: (1.0, 1.5, 0.5),
        min_length: (5, 7, 1),
        max_length: (18, 22, 2),
        smooth_length: (2, 4, 1),
    };

    let cpu = match uma_batch_with_kernel(&prices, Some(&volumes), &sweep, Kernel::ScalarBatch) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(e)),
    };

    let cuda = CudaUma::new(0).expect("CudaUma::new");
    let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
    let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .uma_batch_dev(&prices_f32, Some(&volumes_f32), &sweep)
        .expect("cuda uma_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);
    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda uma batch result to host");

    let tol = 2e-4;
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
fn uma_cuda_one_series_no_volume_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[uma_cuda_one_series_no_volume_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 1536usize;
    let mut prices = vec![f64::NAN; series_len];
    let first_valid = 12usize;
    for i in first_valid..series_len {
        let x = i as f64;
        prices[i] = (x * 0.0013).cos() + 0.0003 * x;
    }

    let sweep = UmaBatchRange {
        accelerator: (1.0, 1.0, 0.0),
        min_length: (6, 8, 1),
        max_length: (20, 24, 2),
        smooth_length: (3, 3, 0),
    };

    let cpu = match uma_batch_with_kernel(&prices, None, &sweep, Kernel::ScalarBatch) {
        Ok(v) => v,
        Err(e) => return Err(Box::new(e)),
    };

    let cuda = CudaUma::new(0).expect("CudaUma::new");
    let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .uma_batch_dev(&prices_f32, None, &sweep)
        .expect("cuda uma_batch_dev (no volume)");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);
    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda uma batch result");

    let tol = 2e-4;
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
fn uma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[uma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 768usize;
    let mut prices_tm = vec![f64::NAN; num_series * series_len];
    let mut volumes_tm = vec![f64::NAN; num_series * series_len];

    for series in 0..num_series {
        let first_valid = 3 + 2 * series;
        for t in first_valid..series_len {
            let x = (t as f64) + (series as f64) * 0.45;
            prices_tm[t * num_series + series] = (x * 0.0021).sin() + 0.0004 * x;
            volumes_tm[t * num_series + series] =
                400.0 + (x * 0.0017).cos() * ((series + 1) as f64) * 30.0;
        }
    }

    let params = UmaParams {
        accelerator: Some(1.3),
        min_length: Some(6),
        max_length: Some(24),
        smooth_length: Some(3),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_prices = vec![f64::NAN; series_len];
        let mut series_volumes = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_prices[t] = prices_tm[t * num_series + series];
            series_volumes[t] = volumes_tm[t * num_series + series];
        }
        let out = UmaBuilder::new()
            .accelerator(params.accelerator.unwrap())
            .min_length(params.min_length.unwrap())
            .max_length(params.max_length.unwrap())
            .smooth_length(params.smooth_length.unwrap())
            .kernel(Kernel::Scalar)
            .apply_slice(&series_prices, Some(&series_volumes))?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = out.values[t];
        }
    }

    let cuda = CudaUma::new(0).expect("CudaUma::new");
    let prices_tm_f32: Vec<f32> = prices_tm.iter().map(|&v| v as f32).collect();
    let volumes_tm_f32: Vec<f32> = volumes_tm.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .uma_many_series_one_param_time_major_dev(
            &prices_tm_f32,
            Some(&volumes_tm_f32),
            num_series,
            series_len,
            &params,
        )
        .expect("cuda uma_many_series_one_param_time_major_dev");

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy uma many-series result to host");

    let tol = 2e-4;
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

#[cfg(feature = "cuda")]
#[test]
fn uma_cuda_many_series_one_param_no_volume_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[uma_cuda_many_series_one_param_no_volume_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let num_series = 3usize;
    let series_len = 640usize;
    let mut prices_tm = vec![f64::NAN; num_series * series_len];

    for series in 0..num_series {
        let first_valid = series + 4;
        for t in first_valid..series_len {
            let x = (t as f64) + 0.37 * (series as f64);
            prices_tm[t * num_series + series] = (x * 0.0019).cos() + 0.00025 * x;
        }
    }

    let params = UmaParams {
        accelerator: Some(1.15),
        min_length: Some(5),
        max_length: Some(20),
        smooth_length: Some(2),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_prices = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_prices[t] = prices_tm[t * num_series + series];
        }
        let out = UmaBuilder::new()
            .accelerator(params.accelerator.unwrap())
            .min_length(params.min_length.unwrap())
            .max_length(params.max_length.unwrap())
            .smooth_length(params.smooth_length.unwrap())
            .kernel(Kernel::Scalar)
            .apply_slice(&series_prices, None)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = out.values[t];
        }
    }

    let cuda = CudaUma::new(0).expect("CudaUma::new");
    let prices_tm_f32: Vec<f32> = prices_tm.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .uma_many_series_one_param_time_major_dev(
            &prices_tm_f32,
            None,
            num_series,
            series_len,
            &params,
        )
        .expect("cuda uma_many_series_one_param_time_major_dev (no volume)");

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy uma many-series result (no volume)");

    let tol = 2e-4;
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
