

use vector_ta::indicators::moving_averages::volatility_adjusted_ma::{
    vama_batch_with_kernel, VamaBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaVama;
#[cfg(feature = "cuda")]
use vector_ta::indicators::moving_averages::volatility_adjusted_ma::{
    vama_with_kernel, VamaInput, VamaParams,
};

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
fn vama_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vama_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00017 * x;
    }

    let sweep = VamaBatchRange {
        base_period: (9, 48, 5),
        vol_period: (5, 21, 4),
    };

    let cuda = CudaVama::new(0).expect("CudaVama::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    
    
    let data32_as_f64: Vec<f64> = data_f32.iter().map(|&v| v as f64).collect();
    let cpu = vama_batch_with_kernel(&data32_as_f64, &sweep, Kernel::ScalarBatch)?;
    let gpu_handle = cuda
        .vama_batch_dev(&data_f32, &sweep)
        .expect("cuda vama_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda vama batch result to host");

    let tol = 2e-5;
    for idx in 0..(cpu.rows * cpu.cols) {
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

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vama_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vama_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 8usize;
    let series_len = 1024usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        for t in series..series_len {
            let x = (t as f64) + (series as f64) * 0.1;
            data_tm[t * num_series + series] = (x * 0.0021).sin() + 0.0003 * x;
        }
    }

    let base_period = 21usize;
    let vol_period = 13usize;
    let params = VamaParams {
        base_period: Some(base_period),
        vol_period: Some(vol_period),
        smoothing: Some(false),
        smooth_type: Some(3),
        smooth_period: Some(5),
    };

    
    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for series in 0..num_series {
        let mut series_data = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series_data[t] = data_tm[t * num_series + series];
        }
        let input = VamaInput::from_slice(&series_data, params.clone());
        let cpu_vals = vama_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + series] = cpu_vals.values[t];
        }
    }

    
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVama::new(0).expect("CudaVama::new");
    let handle = cuda
        .vama_many_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .expect("cuda vama_many_series_one_param_time_major_dev");

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle
        .buf
        .copy_to(&mut gpu_tm)
        .expect("copy cuda vama many-series result to host");

    let tol = 2e-5;
    for idx in 0..gpu_tm.len() {
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
