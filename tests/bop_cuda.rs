// Integration tests for CUDA BOP kernels

use my_project::indicators::bop::{bop_with_kernel, BopInput, BopParams};
use my_project::utilities::data_loader::{read_candles_from_csv, source_type};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaBop;

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
fn bop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 4..len {
        let x = i as f64 * 0.00231;
        open[i] = (x * 0.13).sin() + 0.0007 * x;
        high[i] = open[i] + (0.4 + 0.03 * (x).cos()).abs();
        low[i] = open[i] - (0.39 + 0.02 * (x).sin()).abs();
        close[i] = open[i] + 0.1 * (x * 0.7).sin();
    }

    let cpu = {
        let input = BopInput::from_slices(&open, &high, &low, &close, BopParams::default());
        bop_with_kernel(&input, Kernel::Scalar)?.values
    };

    let open_f32: Vec<f32> = open.iter().map(|&v| v as f32).collect();
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaBop::new(0).expect("CudaBop::new");
    let dev = cuda
        .bop_batch_dev(&open_f32, &high_f32, &low_f32, &close_f32)
        .expect("bop_cuda_batch_dev");

    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-5;
    for i in 0..len {
        assert!(
            approx_eq(cpu[i], host[i] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            cpu[i],
            host[i]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn bop_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bop_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 1024usize; // length
    let mut open_tm = vec![f64::NAN; cols * rows];
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let idx = t * cols + s;
            let x = t as f64 * 0.002 + s as f64 * 0.1;
            let base = (x * 0.31).sin() + 0.0009 * x;
            open_tm[idx] = base + 0.001 * (x).cos();
            high_tm[idx] = base + 0.25 + 0.01 * (x).sin();
            low_tm[idx] = base - 0.24 - 0.01 * (x).cos();
            close_tm[idx] = base + 0.05 * (x * 0.9).sin();
        }
    }

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut o = vec![f64::NAN; rows];
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            o[t] = open_tm[idx];
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
        }
        let input = BopInput::from_slices(&o, &h, &l, &c, BopParams::default());
        let out = bop_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let open_f32: Vec<f32> = open_tm.iter().map(|&v| v as f32).collect();
    let high_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaBop::new(0).expect("CudaBop::new");
    let dev_tm = cuda
        .bop_many_series_one_param_time_major_dev(
            &open_f32, &high_f32, &low_f32, &close_f32, cols, rows,
        )
        .expect("bop_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 5e-5;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}
