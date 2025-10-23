// Integration tests for CUDA IFT RSI kernels

use my_project::indicators::ift_rsi::{
    ift_rsi_batch_with_kernel, ift_rsi_with_kernel, IftRsiBatchRange, IftRsiBuilder, IftRsiInput,
    IftRsiParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaIftRsi;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
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
fn ift_rsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ift_rsi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut data = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = IftRsiBatchRange { rsi_period: (5, 21, 2), wma_period: (9, 21, 2) };

    let cpu = ift_rsi_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaIftRsi::new(0).expect("CudaIftRsi::new");
    let (dev, _combos) = cuda
        .ift_rsi_batch_dev(&data_f32, &sweep)
        .expect("ift_rsi_batch_dev");

    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    // FP32 compensated path on GPU differs from CPU f64; allow headroom for batch.
    let tol = 3e-2;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu.values[idx], host[idx] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            idx,
            cpu.values[idx],
            host[idx]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn ift_rsi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[ift_rsi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1024usize;
    let mut data_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows { // stagger validity per series
            let x = (r as f64) + (s as f64) * 0.2;
            data_tm[r * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }

    let rsi_p = 5usize;
    let wma_p = 9usize;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for r in 0..rows { series[r] = data_tm[r * cols + s]; }
        let out = IftRsiBuilder::new()
            .rsi_period(rsi_p)
            .wma_period(wma_p)
            .apply_slice(&series)?
            .values;
        for r in 0..rows { cpu_tm[r * cols + s] = out[r]; }
    }

    let data_tm_f32: Vec<f32> = data_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaIftRsi::new(0).expect("CudaIftRsi::new");
    let params = IftRsiParams { rsi_period: Some(rsi_p), wma_period: Some(wma_p) };
    let dev_tm = cuda
        .ift_rsi_many_series_one_param_time_major_dev(&data_tm_f32, cols, rows, &params)
        .expect("ift_rsi_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut host_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut host_tm)?;

    let tol = 7e-3;
    for idx in 0..host_tm.len() {
        assert!(approx_eq(cpu_tm[idx], host_tm[idx] as f64, tol), "mismatch at {}", idx);
    }

    Ok(())
}
