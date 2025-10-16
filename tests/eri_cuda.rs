// Integration tests for CUDA ERI kernels

use my_project::indicators::eri::{eri_batch_with_kernel, eri_with_kernel, EriBatchRange, EriData, EriInput, EriParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::eri_wrapper::CudaEri;

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
fn eri_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() { eprintln!("[eri_cuda_batch_matches_cpu] skipped - no CUDA device"); return Ok(()); }

    let len = 8192usize;
    let mut src = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64;
        src[i] = (x * 0.00123).sin() + 0.00017 * x;
        let off = (0.003 * (x * 0.01).sin()).abs() + 0.2;
        high[i] = src[i] + off;
        low[i] = src[i] - off;
    }
    let sweep = EriBatchRange { period: (8, 64, 8), ma_type: "ema".to_string() };
    let cpu = eri_batch_with_kernel(&high, &low, &src, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let src_f32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
    let cuda = CudaEri::new(0).expect("CudaEri::new");
    let ((bull_dev, bear_dev), _combos) = cuda
        .eri_batch_dev(&high_f32, &low_f32, &src_f32, &sweep)
        .expect("cuda eri_batch_dev");

    assert_eq!(cpu.rows, bull_dev.rows);
    assert_eq!(cpu.cols, bull_dev.cols);
    assert_eq!(cpu.rows, bear_dev.rows);
    assert_eq!(cpu.cols, bear_dev.cols);

    let mut bull_host = vec![0f32; bull_dev.len()];
    bull_dev.buf.copy_to(&mut bull_host)?;
    let mut bear_host = vec![0f32; bear_dev.len()];
    bear_dev.buf.copy_to(&mut bear_host)?;

    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(approx_eq(cpu.bull[idx], bull_host[idx] as f64, tol), "bull mismatch at {}", idx);
        assert!(approx_eq(cpu.bear[idx], bear_host[idx] as f64, tol), "bear mismatch at {}", idx);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn eri_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() { eprintln!("[eri_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"); return Ok(()); }

    let cols = 8usize; let rows = 1024usize;
    let mut src_tm = vec![f64::NAN; cols * rows];
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols { for t in s..rows {
        let x = (t as f64) + (s as f64) * 0.2;
        let v = (x * 0.002).sin() + 0.0003 * x;
        src_tm[t*cols + s] = v;
        let off = (0.003 * (x * 0.01).cos()).abs() + 0.2;
        high_tm[t*cols + s] = v + off;
        low_tm[t*cols + s] = v - off;
    }}
    let period = 14usize; let ma_type = "ema";

    // CPU baseline per series
    let mut bull_cpu = vec![f64::NAN; cols * rows];
    let mut bear_cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut z = vec![f64::NAN; rows];
        for t in 0..rows { let idx = t*cols+s; h[t]=high_tm[idx]; l[t]=low_tm[idx]; z[t]=src_tm[idx]; }
        let input = EriInput { data: EriData::Slices { high: &h, low: &l, source: &z }, params: EriParams { period: Some(period), ma_type: Some(ma_type.to_string()) } };
        let out = eri_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows { let idx = t*cols+s; bull_cpu[idx] = out.bull[t]; bear_cpu[idx] = out.bear[t]; }
    }

    // GPU
    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let src_tm_f32: Vec<f32> = src_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaEri::new(0).expect("CudaEri::new");
    let (bull_dev_tm, bear_dev_tm) = cuda
        .eri_many_series_one_param_time_major_dev(&high_tm_f32, &low_tm_f32, &src_tm_f32, cols, rows, period, ma_type)
        .expect("eri_many_series_one_param_time_major_dev");

    assert_eq!(bull_dev_tm.rows, rows); assert_eq!(bull_dev_tm.cols, cols);
    assert_eq!(bear_dev_tm.rows, rows); assert_eq!(bear_dev_tm.cols, cols);

    let mut g_bull_tm = vec![0f32; bull_dev_tm.len()];
    let mut g_bear_tm = vec![0f32; bear_dev_tm.len()];
    bull_dev_tm.buf.copy_to(&mut g_bull_tm)?;
    bear_dev_tm.buf.copy_to(&mut g_bear_tm)?;

    let tol = 1e-4;
    for idx in 0..g_bull_tm.len() {
        assert!(approx_eq(bull_cpu[idx], g_bull_tm[idx] as f64, tol), "bull mismatch at {}", idx);
        assert!(approx_eq(bear_cpu[idx], g_bear_tm[idx] as f64, tol), "bear mismatch at {}", idx);
    }

    Ok(())
}

