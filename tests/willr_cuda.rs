// Integration tests for CUDA WILLR kernels

use my_project::indicators::willr::{
    willr_with_kernel, WillrBatchBuilder, WillrBatchRange, WillrBuilder, WillrInput, WillrParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaWillr;

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
fn willr_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[willr_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let series_len = 4096usize;
    let mut high = vec![f64::NAN; series_len];
    let mut low = vec![f64::NAN; series_len];
    let mut close = vec![f64::NAN; series_len];

    for i in 5..series_len {
        let x = i as f64;
        let base = (x * 0.002).sin() + 0.001 * x;
        high[i] = base + 0.75;
        low[i] = base - 0.65;
        close[i] = base;
    }

    let sweep = WillrBatchRange { period: (9, 48, 3) };

    let cpu = WillrBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slices(&high, &low, &close)?;

    let cuda = CudaWillr::new(0).expect("CudaWillr::new");
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();

    let gpu_handle = cuda
        .willr_batch_dev(&high_f32, &low_f32, &close_f32, &sweep)
        .expect("cuda willr_batch_dev");

    assert_eq!(cpu.rows, gpu_handle.rows);
    assert_eq!(cpu.cols, gpu_handle.cols);

    let mut gpu_host = vec![0f32; gpu_handle.len()];
    gpu_handle
        .buf
        .copy_to(&mut gpu_host)
        .expect("copy cuda willr batch result to host");

    let tol = 5e-5;
    for idx in 0..(cpu.rows * cpu.cols) {
        let cpu_val = cpu.values[idx];
        let gpu_val = gpu_host[idx] as f64;
        assert!(
            approx_eq(cpu_val, gpu_val, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu_val,
            gpu_val
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn willr_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[willr_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 12usize;
    let rows = 2048usize;
    let period = 14usize;

    // Build time-major high/low/close with early NaNs per series
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let fv = s.min(5); // stagger first-valid per series
        for t in fv..rows {
            let x = (t as f64) * 0.0021 + (s as f64) * 0.017;
            let base = (x).sin() + 0.0007 * (t as f64);
            high_tm[t * cols + s] = base + 0.6;
            low_tm[t * cols + s] = base - 0.5;
            close_tm[t * cols + s] = base;
        }
    }

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
        let params = WillrParams {
            period: Some(period),
        };
        let input = WillrInput::from_slices(&h, &l, &c, params);
        let out = willr_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    // GPU
    let high_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaWillr::new(0).expect("CudaWillr::new");
    let dev = cuda
        .willr_many_series_one_param_time_major_dev(
            &high_f32, &low_f32, &close_f32, cols, rows, period,
        )
        .expect("willr many-series dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-5;
    for idx in 0..host.len() {
        let g = host[idx] as f64;
        let s = cpu_tm[idx];
        assert!(
            approx_eq(s, g, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            s,
            g
        );
    }
    Ok(())
}
