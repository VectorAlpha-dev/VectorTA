// Integration tests for CUDA STC kernels (batch and many-series)

use my_project::indicators::stc::{StcBatchBuilder, StcBatchRange, StcBuilder, StcInput, StcParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaStc;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop_stc() {
    #[cfg(not(feature = "cuda"))]
    { assert!(true); }
}

#[cfg(feature = "cuda")]
#[test]
fn stc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 60..len { let x = i as f64; price[i] = (x * 0.00123).sin() + 0.00017 * x; }

    let sweep = StcBatchRange { fast_period: (10, 20, 5), slow_period: (30, 60, 10), k_period: (10, 10, 0), d_period: (3, 3, 0) };

    let cpu = StcBatchBuilder::new().kernel(Kernel::ScalarBatch).apply_slice(&price)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaStc::new(0).expect("CudaStc::new");
    let (dev, _combos) = cuda.stc_batch_dev(&price_f32, &sweep).expect("stc_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut g = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g)?;

    let tol = 1.5e-3; // conservative for cascaded EMA + stochastic on fp32
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let gg = g[idx] as f64;
        assert!(approx_eq(c, gg, tol), "mismatch at {}: cpu={} gpu={}", idx, c, gg);
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn stc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // number of series
    let rows = 2048usize; // time length
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols { for t in 40..rows { let x = (t as f64) + (s as f64)*0.1; tm[t*cols + s] = (x*0.0019).sin() + 0.00011 * x; } }

    // CPU baseline per series
    let params = StcParams { fast_period: Some(23), slow_period: Some(50), k_period: Some(10), d_period: Some(3), fast_ma_type: Some("ema".to_string()), slow_ma_type: Some("ema".to_string()) };
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = tm[t * cols + s]; }
        let input = StcInput::from_slice(&series, params.clone());
        let out = StcBuilder::new().apply_slice(&series)?;
        for t in 0..rows { cpu_tm[t * cols + s] = out.values[t]; }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaStc::new(0).expect("CudaStc::new");
    let dev_tm = cuda.stc_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params).expect("stc many-series");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 2e-3;
    for idx in 0..g_tm.len() {
        assert!(approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol), "mismatch at {}", idx);
    }

    Ok(())
}

