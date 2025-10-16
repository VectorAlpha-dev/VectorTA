// Integration tests for CUDA SRSI kernels

use my_project::indicators::srsi::{srsi_batch_with_kernel, SrsiBatchRange, SrsiParams, SrsiBuilder};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaSrsi;
#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;

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
fn srsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() { eprintln!("[srsi_cuda_batch_matches_cpu] skipped - no CUDA device"); return Ok(()); }
    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 8..len { let x = i as f64; price[i] = (x * 0.00123).sin() + 0.00031 * x.cos(); }
    let sweep = SrsiBatchRange { rsi_period: (4, 22, 3), stoch_period: (4, 20, 4), k: (3, 5, 1), d: (3, 5, 1) };

    let cpu = srsi_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaSrsi::new(0).expect("CudaSrsi::new");
    let (dev_pair, combos) = cuda.srsi_batch_dev(&price_f32, &sweep).expect("srsi_cuda_batch_dev");
    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(dev_pair.k.rows, cpu.rows);
    assert_eq!(dev_pair.k.cols, cpu.cols);
    assert_eq!(dev_pair.d.rows, cpu.rows);
    assert_eq!(dev_pair.d.cols, cpu.cols);

    let mut gk = vec![0f32; dev_pair.k.len()];
    let mut gd = vec![0f32; dev_pair.d.len()];
    dev_pair.k.buf.copy_to(&mut gk)?;
    dev_pair.d.buf.copy_to(&mut gd)?;
    let tol = 8e-4;
    for idx in 0..gk.len() {
        assert!(approx_eq(cpu.k[idx], gk[idx] as f64, tol), "K mismatch at {}", idx);
        assert!(approx_eq(cpu.d[idx], gd[idx] as f64, tol), "D mismatch at {}", idx);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn srsi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() { eprintln!("[srsi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"); return Ok(()); }
    let cols = 8usize; let rows = 2048usize;
    let mut prices_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols { for t in s..rows { let idx = t * cols + s; let x = (t as f64) * 0.002 + (s as f64) * 0.01; prices_tm[idx] = (x.sin()*0.7 + x*0.0009).into(); } }
    let rp = 14usize; let sp = 14usize; let kp = 3usize; let dp = 3usize;

    // CPU baseline per series
    let mut cpu_k = vec![f64::NAN; cols * rows];
    let mut cpu_d = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = prices_tm[t * cols + s]; }
        let params = SrsiParams { rsi_period: Some(rp), stoch_period: Some(sp), k: Some(kp), d: Some(dp), source: None };
        let out = SrsiBuilder::new()
            .rsi_period(rp).stoch_period(sp).k(kp).d(dp)
            .apply_slice(&series)?;
        for t in 0..rows { cpu_k[t * cols + s] = out.k[t]; cpu_d[t * cols + s] = out.d[t]; }
    }

    let prices_tm_f32: Vec<f32> = prices_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaSrsi::new(0).expect("CudaSrsi::new");
    let dev_pair = cuda
        .srsi_many_series_one_param_time_major_dev(&prices_tm_f32, cols, rows, &SrsiParams { rsi_period: Some(rp), stoch_period: Some(sp), k: Some(kp), d: Some(dp), source: None })
        .expect("srsi many-series");

    assert_eq!(dev_pair.k.rows, rows); assert_eq!(dev_pair.k.cols, cols);
    assert_eq!(dev_pair.d.rows, rows); assert_eq!(dev_pair.d.cols, cols);
    let mut gk = vec![0f32; dev_pair.k.len()]; let mut gd = vec![0f32; dev_pair.d.len()];
    dev_pair.k.buf.copy_to(&mut gk)?; dev_pair.d.buf.copy_to(&mut gd)?;
    let tol = 1e-3;
    for i in 0..gk.len() { assert!(approx_eq(cpu_k[i], gk[i] as f64, tol), "K mismatch at {}", i); assert!(approx_eq(cpu_d[i], gd[i] as f64, tol), "D mismatch at {}", i); }
    Ok(())
}
