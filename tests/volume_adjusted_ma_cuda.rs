// Integration tests for CUDA Volume Adjusted Moving Average (VAMA) kernels

use my_project::indicators::moving_averages::volume_adjusted_ma::{
    VolumeAdjustedMaBatchRange, VolumeAdjustedMaBuilder, VolumeAdjustedMaParams,
    VolumeAdjustedMa_batch_with_kernel,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaVolumeAdjustedMa;

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
fn volume_adjusted_ma_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[volume_adjusted_ma_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut prices = vec![f64::NAN; series_len];
    let mut volumes = vec![f64::NAN; series_len];
    for i in 12..series_len {
        let x = i as f64;
        prices[i] = (x * 0.004).sin() + 0.0004 * x;
        volumes[i] = ((x * 0.006).cos().abs() + 1.5) * 750.0;
    }

    let sweep = VolumeAdjustedMaBatchRange {
        length: (5, 21, 4),
        vi_factor: (0.45, 1.05, 0.2),
        sample_period: (0, 12, 4),
        strict: None,
    };

    let cpu = VolumeAdjustedMa_batch_with_kernel(&prices, &volumes, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaVolumeAdjustedMa::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
    let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .vama_batch_dev(&prices_f32, &volumes_f32, &sweep)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    let tol = 1e-4;
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
fn volume_adjusted_ma_cuda_batch_strict_only_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[volume_adjusted_ma_cuda_batch_strict_only_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let series_len = 3072usize;
    let mut prices = vec![f64::NAN; series_len];
    let mut volumes = vec![f64::NAN; series_len];
    for i in 16..series_len {
        let x = i as f64;
        prices[i] = (x * 0.0025).cos() + 0.00025 * x;
        volumes[i] = ((x * 0.01).sin().abs() + 0.8) * 1200.0;
    }

    let sweep = VolumeAdjustedMaBatchRange {
        length: (9, 33, 6),
        vi_factor: (0.55, 0.95, 0.15),
        sample_period: (0, 0, 0),
        strict: Some(true),
    };

    let cpu = VolumeAdjustedMa_batch_with_kernel(&prices, &volumes, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaVolumeAdjustedMa::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let prices_f32: Vec<f32> = prices.iter().map(|&v| v as f32).collect();
    let volumes_f32: Vec<f32> = volumes.iter().map(|&v| v as f32).collect();
    let gpu_handle = cuda
        .vama_batch_dev(&prices_f32, &volumes_f32, &sweep)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    let tol = 1e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "strict mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn volume_adjusted_ma_cuda_many_series_one_param_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[volume_adjusted_ma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 1536usize;
    let length = 18usize;
    let vi_factor = 0.72f64;
    let sample_period = 12usize;
    let strict = false;

    let mut prices_tm = vec![f64::NAN; num_series * series_len];
    let mut volume_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in (series + 4)..series_len {
            let idx = t * num_series + series;
            let base = t as f64 + (series as f64) * 0.25;
            prices_tm[idx] = (base * 0.003).sin() + 0.0003 * base;
            volume_tm[idx] = ((base * 0.009).cos().abs() + 1.1) * (450.0 + series as f64 * 35.0);
        }
    }

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut price_series = vec![f64::NAN; series_len];
        let mut volume_series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            let idx = t * num_series + series;
            price_series[t] = prices_tm[idx];
            volume_series[t] = volume_tm[idx];
        }
        let cpu_result = VolumeAdjustedMaBuilder::new()
            .length(length)
            .vi_factor(vi_factor)
            .strict(strict)
            .sample_period(sample_period)
            .kernel(Kernel::Scalar)
            .apply_slices(&price_series, &volume_series)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = cpu_result.values[t];
        }
    }

    let cuda = CudaVolumeAdjustedMa::new(0).map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    let price_tm_f32: Vec<f32> = prices_tm.iter().map(|&v| v as f32).collect();
    let volume_tm_f32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();
    let params = VolumeAdjustedMaParams {
        length: Some(length),
        vi_factor: Some(vi_factor),
        strict: Some(strict),
        sample_period: Some(sample_period),
    };
    let gpu_handle = cuda
        .vama_multi_series_one_param_time_major_dev(
            &price_tm_f32,
            &volume_tm_f32,
            num_series,
            series_len,
            &params,
        )
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    assert_eq!(gpu_handle.rows, series_len);
    assert_eq!(gpu_handle.cols, num_series);

    let mut gpu_tm = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_tm)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    let tol = 1e-4;
    for idx in 0..(num_series * series_len) {
        let a = cpu_tm[idx];
        let b = gpu_tm[idx] as f64;
        assert!(
            approx_eq(a, b, tol),
            "many-series mismatch at {}: cpu={} gpu={}",
            idx,
            a,
            b
        );
    }

    Ok(())
}
