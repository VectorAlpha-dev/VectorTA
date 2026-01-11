

use vector_ta::indicators::moving_averages::sma::{
    sma_batch_with_kernel, SmaBatchRange, SmaBuilder, SmaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaSma;

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
fn sma_cuda_one_series_many_params_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sma_cuda_one_series_many_params_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut data = vec![f64::NAN; series_len];
    for i in 5..series_len {
        let x = i as f64;
        data[i] = (x * 0.001).sin() + 0.0002 * x;
    }

    let sweep = SmaBatchRange { period: (5, 45, 5) };

    let cpu = sma_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let cuda = CudaSma::new(0).expect("CudaSma::new");
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let (dev, combos) = cuda
        .sma_batch_dev(&data_f32, &sweep)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(dev.cols, cpu.cols);
    assert_eq!(dev.rows, cpu.rows);

    for (idx, combo) in combos.iter().enumerate() {
        assert_eq!(combo.period, cpu.combos[idx].period);
    }

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let tol = 1e-5;
    for i in 0..gpu_flat.len() {
        let a = cpu.values[i];
        let b = gpu_flat[i] as f64;
        assert!(
            approx_eq(a, b, tol),
            "Mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn sma_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[sma_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 4usize;
    let series_len = 2048usize;
    let mut data_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        for t in (j + 2)..series_len {
            let x = t as f64 + j as f64 * 0.25;
            data_tm[t * num_series + j] = (x * 0.002).cos() + 0.0003 * x;
        }
    }

    let period = 14usize;

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            series[t] = data_tm[t * num_series + j];
        }
        let out = SmaBuilder::new()
            .period(period)
            .apply_slice(&series)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    let cuda = CudaSma::new(0).expect("CudaSma::new");
    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let params = SmaParams {
        period: Some(period),
    };
    let dev = cuda
        .sma_multi_series_one_param_time_major_dev(&data_tm_f32, num_series, series_len, &params)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    assert_eq!(dev.rows, series_len);
    assert_eq!(dev.cols, num_series);

    let mut gpu_flat = vec![0f32; dev.len()];
    dev.buf
        .copy_to(&mut gpu_flat)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    let tol = 1e-5;
    for i in 0..gpu_flat.len() {
        let a = cpu_tm[i];
        let b = gpu_flat[i] as f64;
        assert!(
            approx_eq(a, b, tol),
            "Mismatch at {}: cpu={} gpu={}",
            i,
            a,
            b
        );
    }

    Ok(())
}
