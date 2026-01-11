

use vector_ta::indicators::moving_averages::tradjema::{
    tradjema_with_kernel, TradjemaBatchRange, TradjemaInput, TradjemaParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::moving_averages::CudaTradjema;

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
fn tradjema_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[tradjema_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut high = vec![f64::NAN; series_len];
    let mut low = vec![f64::NAN; series_len];
    let mut close = vec![f64::NAN; series_len];

    for i in 8..series_len {
        let t = i as f64;
        let base = (t * 0.0023).sin() + 0.00035 * t;
        close[i] = base;
        high[i] = base + 0.25 + 0.01 * (i % 7) as f64;
        low[i] = base - 0.27 - 0.005 * (i % 5) as f64;
    }

    let params = TradjemaParams {
        length: Some(36),
        mult: Some(8.5),
    };

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let high_quant: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
    let low_quant: Vec<f64> = low_f32.iter().map(|&v| v as f64).collect();
    let close_quant: Vec<f64> = close_f32.iter().map(|&v| v as f64).collect();

    let input = TradjemaInput::from_slices(&high_quant, &low_quant, &close_quant, params.clone());
    let cpu = tradjema_with_kernel(&input, Kernel::Scalar)?;

    let sweep = TradjemaBatchRange {
        length: (params.length.unwrap(), params.length.unwrap(), 0),
        mult: (params.mult.unwrap(), params.mult.unwrap(), 0.0),
    };

    let cuda = CudaTradjema::new(0)?;
    let handle = cuda.tradjema_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)?;

    assert_eq!(handle.rows, 1);
    assert_eq!(handle.cols, series_len);

    let mut gpu_host = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_host)?;

    let tol = 1e-4;
    for idx in 0..series_len {
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
fn tradjema_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[tradjema_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let num_series = 5usize;
    let series_len = 2048usize;

    let mut high_tm = vec![f64::NAN; num_series * series_len];
    let mut low_tm = vec![f64::NAN; num_series * series_len];
    let mut close_tm = vec![f64::NAN; num_series * series_len];

    for j in 0..num_series {
        for t in (j + 10)..series_len {
            let base = (t as f64) * 0.0015 + 0.02 * j as f64;
            let osc = (t as f64 * 0.004 + j as f64 * 0.2).sin();
            let close_val = base + osc * 0.6;
            let high_val = close_val + 0.22 + 0.01 * (j as f64);
            let low_val = close_val - 0.24 - 0.007 * (j as f64);

            let idx = t * num_series + j;
            close_tm[idx] = close_val;
            high_tm[idx] = high_val;
            low_tm[idx] = low_val;
        }
    }

    let params = TradjemaParams {
        length: Some(30),
        mult: Some(7.0),
    };

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();

    let high_tm_quant: Vec<f64> = high_tm_f32.iter().map(|&v| v as f64).collect();
    let low_tm_quant: Vec<f64> = low_tm_f32.iter().map(|&v| v as f64).collect();
    let close_tm_quant: Vec<f64> = close_tm_f32.iter().map(|&v| v as f64).collect();

    let mut cpu_tm = vec![f64::NAN; num_series * series_len];
    for j in 0..num_series {
        let mut high_series = vec![f64::NAN; series_len];
        let mut low_series = vec![f64::NAN; series_len];
        let mut close_series = vec![f64::NAN; series_len];
        for t in 0..series_len {
            let idx = t * num_series + j;
            high_series[t] = high_tm_quant[idx];
            low_series[t] = low_tm_quant[idx];
            close_series[t] = close_tm_quant[idx];
        }
        let input =
            TradjemaInput::from_slices(&high_series, &low_series, &close_series, params.clone());
        let out = tradjema_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..series_len {
            cpu_tm[t * num_series + j] = out.values[t];
        }
    }

    let cuda = CudaTradjema::new(0)?;
    let handle = cuda.tradjema_many_series_one_param_time_major_dev(
        &high_tm_f32,
        &low_tm_f32,
        &close_tm_f32,
        num_series,
        series_len,
        &params,
    )?;

    assert_eq!(handle.rows, series_len);
    assert_eq!(handle.cols, num_series);

    let mut gpu_tm = vec![0f32; handle.len()];
    handle.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-4;
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
