// Integration tests for CUDA Ehlers ITrend kernels

use my_project::indicators::moving_averages::ehlers_itrend::{
    ehlers_itrend_batch_with_kernel, ehlers_itrend_with_kernel, EhlersITrendBatchRange,
    EhlersITrendInput, EhlersITrendParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaEhlersITrend;

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
fn ehlers_itrend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ehlers_itrend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 0..series_len {
        let x = i as f64 * 0.013 + (i % 17) as f64 * 0.21;
        data[i] = (x * 0.0017).sin() + 0.00031 * x.cos();
    }

    let sweep = EhlersITrendBatchRange {
        warmup_bars: (8, 16, 4),
        max_dc_period: (30, 50, 10),
    };

    let cpu = ehlers_itrend_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaEhlersITrend::new(0).expect("CudaEhlersITrend::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let handle = cuda
        .ehlers_itrend_batch_dev(&data_f32, &sweep)
        .expect("ehlers_itrend cuda batch dev");

    assert_eq!(cpu.rows, handle.rows);
    assert_eq!(cpu.cols, handle.cols);

    let mut gpu_flat = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_flat)
        .expect("copy itrend cuda batch result to host");

    let tol = 5e-4;
    for idx in 0..cpu.values.len() {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_flat[idx] as f64;
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
fn ehlers_itrend_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[ehlers_itrend_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 2048usize;
    let warmup_bars = 12usize;
    let max_dc_period = 48usize;

    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in 0..series_len {
            let base = (t as f64) * 0.73 + (series as f64) * 1.17;
            data_tm[t * num_series + series] = (base * 0.0013).cos() + 0.0021 * base.sin();
        }
    }

    let params = EhlersITrendParams {
        warmup_bars: Some(warmup_bars),
        max_dc_period: Some(max_dc_period),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut column = vec![f64::NAN; series_len];
        for t in 0..series_len {
            column[t] = data_tm[t * num_series + series];
        }
        let input = EhlersITrendInput::from_slice(&column, params.clone());
        let cpu_vals = ehlers_itrend_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = cpu_vals.values[t];
        }
    }

    let cuda = CudaEhlersITrend::new(0).expect("CudaEhlersITrend::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let handle = cuda
        .ehlers_itrend_many_series_one_param_time_major_dev(
            &data_tm_f32,
            num_series,
            series_len,
            &params,
        )
        .expect("ehlers_itrend cuda many-series dev");

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy itrend cuda many-series result to host");

    let tol = 5e-4;
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
