// CUDA integration tests for DevStop

use my_project::indicators::devstop::{
    devstop_batch_with_kernel, devstop_with_kernel, DevStopBatchRange, DevStopData, DevStopInput,
    DevStopParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaDevStop;

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
fn devstop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[devstop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 20_000usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        let c = (x * 0.0019).sin() + 0.00021 * x;
        let off = (0.35 + 0.01 * (x * 0.00031).cos().abs());
        high[i] = c + off;
        low[i] = c - off;
    }
    let sweep = DevStopBatchRange {
        period: (10, 30, 5),
        mult: (0.0, 2.0, 0.5),
        devtype: (0, 0, 0),
    };

    let cpu = devstop_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;

    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaDevStop::new(0).expect("CudaDevStop::new");
    let (dev, meta) = cuda
        .devstop_batch_dev(&high_f32, &low_f32, &sweep, true)
        .expect("devstop batch");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(cpu.rows, meta.len());

    let mut host_out = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host_out)?;

    // Compare row × col with modest tolerance
    let tol = 5e-2; // FP32 path
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host_out[idx] as f64;
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
fn devstop_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[devstop_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 16usize;
    let rows = 8192usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.73;
            let c = (x * 0.0021).sin() + 0.00017 * x;
            let off = 0.25 + 0.01 * (x * 0.0017).cos().abs();
            high_tm[t * cols + s] = c + off;
            low_tm[t * cols + s] = c - off;
        }
    }
    let period = 20usize;
    let mult = 1.5f64;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
        }
        let params = DevStopParams {
            period: Some(period),
            mult: Some(mult),
            devtype: Some(0),
            direction: Some("long".into()),
            ma_type: Some("sma".into()),
        };
        let input = DevStopInput {
            data: DevStopData::SliceHL(&h, &l),
            params,
        };
        let out = devstop_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let high_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaDevStop::new(0).expect("CudaDevStop::new");
    let dev = cuda
        .devstop_many_series_one_param_time_major_dev(
            &high_f32,
            &low_f32,
            cols,
            rows,
            period,
            mult as f32,
            true,
        )
        .expect("devstop many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-2;
    for i in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[i], host[i] as f64, tol),
            "mismatch at {}",
            i
        );
    }
    Ok(())
}
