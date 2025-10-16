// CUDA LRSI tests: compare GPU vs CPU for batch and many-series

use my_project::indicators::lrsi::{
    lrsi, lrsi_batch_with_kernel, LrsiBatchRange, LrsiData, LrsiInput, LrsiParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaLrsi;
#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop() { #[cfg(not(feature = "cuda"))] assert!(true); }

#[cfg(feature = "cuda")]
#[test]
fn lrsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[lrsi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 32768usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        let mid = (x * 0.00123).sin() + 0.00013 * x;
        let off = (x * 0.00077).cos().abs() + 0.2;
        high[i] = mid + off;
        low[i] = mid - off;
    }
    let sweep = LrsiBatchRange { alpha: (0.1, 0.9, 0.2) };
    let cpu = lrsi_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let mut cuda = CudaLrsi::new(0).expect("CudaLrsi::new");
    let dev = cuda
        .lrsi_batch_dev(&high_f32, &low_f32, &sweep)
        .expect("lrsi_batch_dev");
    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(approx_eq(c, g, tol), "mismatch at {}: cpu={} gpu={}", idx, c, g);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn lrsi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[lrsi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 16usize;
    let rows = 16384usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.5;
            let mid = (x * 0.0021).sin() + 0.0002 * x;
            let off = (x * 0.0014).cos().abs() + 0.15;
            let idx = t * cols + s;
            high_tm[idx] = mid + off;
            low_tm[idx] = mid - off;
        }
    }
    let alpha = 0.2;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut high = vec![f64::NAN; rows];
        let mut low = vec![f64::NAN; rows];
        for t in 0..rows { let idx = t * cols + s; high[t] = high_tm[idx]; low[t] = low_tm[idx]; }
        let inp = LrsiInput { data: LrsiData::Slices { high: &high, low: &low }, params: LrsiParams { alpha: Some(alpha) } };
        let out = lrsi(&inp)?.values;
        for t in 0..rows { cpu_tm[t * cols + s] = out[t]; }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let mut cuda = CudaLrsi::new(0).expect("CudaLrsi::new");
    let dev = cuda
        .lrsi_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, cols, rows, alpha)
        .expect("lrsi_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1e-4;
    for i in 0..host.len() {
        assert!(approx_eq(cpu_tm[i], host[i] as f64, tol), "mismatch at {}", i);
    }
    Ok(())
}

