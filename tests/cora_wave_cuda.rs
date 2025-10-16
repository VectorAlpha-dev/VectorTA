// Integration tests for CUDA CoRa Wave kernels

use my_project::indicators::cora_wave::{
    cora_wave_batch_with_kernel, CoraWaveBatchRange, CoraWaveBuilder, CoraWaveParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaCoraWave;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    { assert!(true); }
}

#[cfg(feature = "cuda")]
#[test]
fn cora_wave_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cora_wave_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 6..series_len {
        let x = i as f64;
        data[i] = (x * 0.0015).sin() + 0.0003 * x;
    }

    let sweep = CoraWaveBatchRange { period: (8, 40, 4), r_multi: (1.5, 2.0, 0.5), smooth: true };
    let cpu = cora_wave_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaCoraWave::new(0)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda.cora_wave_batch_dev(&data_f32, &sweep)?;

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);
    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut gpu_host)?;

    let tol = 5e-3; // smoothing + geometric weights → allow slightly looser
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
        assert!(approx_eq(a, b, tol), "mismatch at {}: cpu={} gpu={}", idx, a, b);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cora_wave_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cora_wave_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 8usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in (j + 3)..series_len {
            let x = (t as f64) + (j as f64) * 0.07;
            data_tm[t * num_series + j] = (x * 0.0012).cos() + 0.00025 * x;
        }
    }
    let params = CoraWaveParams { period: Some(24), r_multi: Some(2.0), smooth: Some(true) };

    // CPU reference per series
    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len { series[t] = data_tm[t * num_series + j]; }
        let out = CoraWaveBuilder::default()
            .period(params.period.unwrap())
            .r_multi(params.r_multi.unwrap())
            .smooth(params.smooth.unwrap())
            .apply_slice(&series)?;
        for t in 0..series_len { cpu_tm[t * num_series + j] = out.values[t]; }
    }

    let cuda = CudaCoraWave::new(0)?;
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let gpu = cuda.cora_wave_multi_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)?;

    assert_eq!(gpu.rows, series_len);
    assert_eq!(gpu.cols, num_series);
    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut gpu_host)?;

    let tol = 6e-3;
    for idx in 0..(num_series * series_len) {
        let a = cpu_tm[idx];
        let b = gpu_host[idx] as f64;
        assert!(approx_eq(a, b, tol), "mismatch at {}: cpu={} gpu={}", idx, a, b);
    }
    Ok(())
}

