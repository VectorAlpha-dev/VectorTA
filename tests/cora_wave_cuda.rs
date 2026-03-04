use vector_ta::indicators::cora_wave::{
    cora_wave_batch_with_kernel, CoraWaveBatchRange, CoraWaveBuilder, CoraWaveParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::{CudaCoraWave, CudaMaData, CudaMaSelector};

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

    let sweep = CoraWaveBatchRange {
        period: (8, 40, 4),
        r_multi: (1.5, 2.0, 0.5),
        smooth: true,
    };
    let cpu = cora_wave_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaCoraWave::new(0)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let gpu = cuda.cora_wave_batch_dev(&data_f32, &sweep)?;

    assert_eq!(cpu.rows, gpu.rows);
    assert_eq!(cpu.cols, gpu.cols);
    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut gpu_host)?;

    let tol = 5e-3;
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
    let params = CoraWaveParams {
        period: Some(24),
        r_multi: Some(2.0),
        smooth: Some(true),
    };

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = CoraWaveBuilder::default()
            .period(params.period.unwrap())
            .r_multi(params.r_multi.unwrap())
            .smooth(params.smooth.unwrap())
            .apply_slice(&series)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    let cuda = CudaCoraWave::new(0)?;
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let gpu = cuda.cora_wave_multi_series_one_param_time_major_dev(
        &data_tm_f32,
        num_series,
        series_len,
        &params,
    )?;

    assert_eq!(gpu.rows, series_len);
    assert_eq!(gpu.cols, num_series);
    let mut gpu_host = vec![0f32; gpu.len()];
    gpu.buf.copy_to(&mut gpu_host)?;

    let tol = 6e-3;
    for idx in 0..(num_series * series_len) {
        let a = cpu_tm[idx];
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
fn cora_wave_cuda_selector_dispatch_single_and_sweep() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cora_wave_cuda_selector_dispatch_single_and_sweep] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 2048usize;
    let mut data_f64 = vec![f64::NAN; series_len];
    for i in 7..series_len {
        let x = i as f64;
        data_f64[i] = (x * 0.0014).sin() + 0.00029 * x;
    }
    let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();

    let mut params = HashMap::new();
    params.insert("r_multi".to_string(), 2.0);
    params.insert("smooth".to_string(), 1.0);

    let selector = CudaMaSelector::new(0);
    let single = selector
        .ma_to_device_with_params("cora_wave", CudaMaData::SliceF32(&data_f32), 20, &params)
        .map_err(|e| e.to_string())?;

    assert_eq!(single.rows, 1);
    assert_eq!(single.cols, series_len);

    let sweep = selector
        .ma_sweep_to_device_with_params(
            "cora_wave",
            CudaMaData::SliceF32(&data_f32),
            10,
            18,
            4,
            &params,
        )
        .map_err(|e| e.to_string())?;

    assert_eq!(sweep.rows, 3);
    assert_eq!(sweep.cols, series_len);

    let cuda = CudaCoraWave::new(0)?;
    let direct = cuda.cora_wave_batch_dev(
        &data_f32,
        &CoraWaveBatchRange {
            period: (10, 18, 4),
            r_multi: (2.0, 2.0, 0.0),
            smooth: true,
        },
    )?;

    assert_eq!(sweep.rows, direct.rows);
    assert_eq!(sweep.cols, direct.cols);

    let mut sweep_host = vec![0f32; sweep.len()];
    let mut direct_host = vec![0f32; direct.len()];
    sweep.buf.copy_to(&mut sweep_host)?;
    direct.buf.copy_to(&mut direct_host)?;

    for idx in 0..sweep_host.len() {
        assert!(
            approx_eq(sweep_host[idx] as f64, direct_host[idx] as f64, 1e-4),
            "selector/direct mismatch at {}: selector={} direct={}",
            idx,
            sweep_host[idx],
            direct_host[idx]
        );
    }

    Ok(())
}
