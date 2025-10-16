// CUDA integration tests for Kaufman Efficiency Ratio (ER)

use my_project::utilities::enums::Kernel;
use my_project::indicators::er::{er_batch_with_kernel, er_with_kernel, ErBatchRange, ErInput, ErParams, ErData};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::er_wrapper::CudaEr;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { true } else { (a - b).abs() <= tol }
}

#[test]
fn cuda_feature_off_noop() { #[cfg(not(feature = "cuda"))] assert!(true); }

#[cfg(feature = "cuda")]
#[test]
fn er_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[er_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 6..len { let x = i as f64; price[i] = (x * 0.00123).sin() + 0.00017 * x; }
    let sweep = ErBatchRange { period: (5, 49, 2) };

    // CPU baseline
    let cpu = er_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    // GPU
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaEr::new(0).expect("CudaEr::new");
    let dev = cuda.er_batch_dev(&price_f32, &sweep).expect("er_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;

    let tol = 5e-2; // ER in FP32 can drift up to ~4e-2 vs f64 in worst cases
    for idx in 0..got.len() {
        assert!(approx_eq(cpu.values[idx], got[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn er_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[er_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 1024usize; // time
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols { for t in s..rows { let x = (t as f64) + (s as f64) * 0.2; tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x; } }
    let period = 20usize;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = tm[t * cols + s]; }
        let params = ErParams { period: Some(period) };
        let input = ErInput { data: ErData::Slice(&series), params };
        let out = er_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows { cpu_tm[t * cols + s] = out.values[t]; }
    }

    // GPU
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaEr::new(0).expect("CudaEr::new");
    let dev = cuda
        .er_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period)
        .expect("er_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut got_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got_tm)?;

    let tol = 5e-4;
    for idx in 0..got_tm.len() {
        assert!(approx_eq(cpu_tm[idx], got_tm[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}
