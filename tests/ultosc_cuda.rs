// Integration tests for CUDA UltOsc kernels

use my_project::indicators::ultosc::{
    ultosc_batch_with_kernel, ultosc_with_kernel, UltOscBatchRange, UltOscInput, UltOscParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaUltosc;

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
fn ultosc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ultosc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 32768usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 1..len {
        let x = i as f64;
        // create a plausible candle with some spread
        let base = (x * 0.00123).sin() + 0.00017 * x;
        let spread = (0.0031 * x.cos()).abs() + 0.05;
        close[i] = base;
        high[i] = base + spread;
        low[i] = base - spread;
    }
    let sweep = UltOscBatchRange {
        timeperiod1: (5, 29, 4),
        timeperiod2: (10, 50, 8),
        timeperiod3: (20, 70, 10),
    };

    let cpu = ultosc_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaUltosc::new(0).expect("CudaUltosc::new");
    let dev = cuda
        .ultosc_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("cuda ultosc_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 3e-3; // loosened for FP32 double-single path
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            c,
            g
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ultosc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ultosc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 1..rows {
            // start at 1 so prev-close is present
            let x = (t as f64) + (s as f64) * 0.41;
            let base = (x * 0.002).sin() + 0.0003 * x;
            let spread = (x * 0.0013).cos().abs() + 0.04;
            let idx = t * cols + s;
            close_tm[idx] = base;
            high_tm[idx] = base + spread;
            low_tm[idx] = base - spread;
        }
    }
    let p1 = 7usize;
    let p2 = 14usize;
    let p3 = 28usize;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
        }
        let params = UltOscParams {
            timeperiod1: Some(p1),
            timeperiod2: Some(p2),
            timeperiod3: Some(p3),
        };
        let input = UltOscInput::from_slices(&h, &l, &c, params);
        let out = ultosc_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let hf: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaUltosc::new(0).expect("CudaUltosc::new");
    let dev_tm = cuda
        .ultosc_many_series_one_param_time_major_dev(&hf, &lf, &cf, cols, rows, p1, p2, p3)
        .expect("ultosc many-series");
    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut host_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut host_tm)?;
    let tol = 3e-3; // allow small drift vs CPU in FP32 compensated path
    for idx in 0..host_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], host_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
