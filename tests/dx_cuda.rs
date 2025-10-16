// Integration tests for CUDA DX kernels

use my_project::indicators::dx::{dx_batch_with_kernel, DxBatchRange};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::{cuda_available, CudaDx};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() { #[cfg(not(feature = "cuda"))] { assert!(true); } }

#[cfg(feature = "cuda")]
#[test]
fn dx_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dx_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut close = vec![f64::NAN; len];
    for i in 0..len { if i >= 4 { let x = i as f64; close[i] = (x*0.0023).sin() + 0.0004*x; } }
    let mut high = close.clone();
    let mut low = close.clone();
    for i in 0..len { if !close[i].is_nan() { let x = i as f64*0.0025; let off = (0.002*x.sin()).abs() + 0.15; high[i] = close[i] + off; low[i] = close[i] - off; }}
    let sweep = DxBatchRange { period: (6, 30, 3) };

    // Quantize to f32 to match CUDA input domain
    let high_q: Vec<f64> = high.iter().map(|&v| (v as f32) as f64).collect();
    let low_q: Vec<f64> = low.iter().map(|&v| (v as f32) as f64).collect();
    let close_q: Vec<f64> = close.iter().map(|&v| (v as f32) as f64).collect();
    let cpu = dx_batch_with_kernel(&high_q, &low_q, &close_q, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let cuda = CudaDx::new(0).expect("CudaDx::new");
    let (dev, combos) = cuda
        .dx_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("cuda dx_batch_dev");

    assert_eq!(cpu.rows, combos.len());
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut gpu_host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_host)?;

    let tol = 2e-1; // FP32 tolerance (DX ratio is sensitive; allow 0.2 abs)
    for idx in 0..(cpu.rows * cpu.cols) {
        let a = cpu.values[idx];
        let b = gpu_host[idx] as f64;
        assert!(approx_eq(a, b, tol), "mismatch at {}: cpu={} gpu={}", idx, a, b);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn dx_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dx_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 16usize; // series
    let rows = 2048usize; // time
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols { for t in s..rows { let x = (t as f64) + (s as f64)*0.2; close_tm[t*cols+s] = (x*0.002).sin() + 0.0003*x; }}
    let mut high_tm = close_tm.clone();
    let mut low_tm = close_tm.clone();
    for s in 0..cols { for t in 0..rows { let idx = t*cols + s; if !close_tm[idx].is_nan() { let x = (t as f64)*0.0025; let off = (0.002*x.sin()).abs() + 0.15; high_tm[idx] = close_tm[idx] + off; low_tm[idx] = close_tm[idx] - off; }}}

    let period = 14usize;

    // CPU baseline per series (using batch helper with single param)
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows { h[t] = high_tm[t*cols+s]; l[t] = low_tm[t*cols+s]; c[t] = close_tm[t*cols+s]; }
        let sweep = DxBatchRange { period: (period, period, 0) };
        let hq: Vec<f64> = h.iter().map(|&v| (v as f32) as f64).collect();
        let lq: Vec<f64> = l.iter().map(|&v| (v as f32) as f64).collect();
        let cq: Vec<f64> = c.iter().map(|&v| (v as f32) as f64).collect();
        let out = dx_batch_with_kernel(&hq, &lq, &cq, &sweep, Kernel::ScalarBatch)?;
        let row = 0usize;
        for t in 0..rows { cpu_tm[t*cols+s] = out.values[row*out.cols + t]; }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDx::new(0).expect("CudaDx::new");
    let dev = cuda
        .dx_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, &close_tm_f32, cols, rows, period)
        .expect("dx many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut g_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g_tm)?;
    let tol = 1e-1; // DX is more sensitive than ADX; allow 0.1 abs
    for i in 0..g_tm.len() {
        assert!(approx_eq(cpu_tm[i], g_tm[i] as f64, tol), "mismatch at {}", i);
    }
    Ok(())
}
