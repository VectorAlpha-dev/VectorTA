// Integration tests for CUDA AlphaTrend kernels

use my_project::indicators::alphatrend::{
    alphatrend_batch_slice, AlphaTrendBatchRange, AlphaTrendParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::alphatrend_wrapper::CudaAlphaTrend;

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
fn alphatrend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[alphatrend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16384usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        high[i] = (x * 0.0007).sin() + 0.02;
        low[i] = high[i] - 0.03 - 0.01 * (x * 0.0013).cos().abs();
        close[i] = (high[i] + low[i]) * 0.5 + 0.001 * (x * 0.001).cos();
        volume[i] = (x * 0.0009).cos().abs() + 0.5;
    }
    let sweep = AlphaTrendBatchRange { coeff: (0.9, 1.1, 0.1), period: (10, 30, 10), no_volume: true };

    let open = close.clone();
    let cpu = alphatrend_batch_slice(&open, &high, &low, &close, &volume, &sweep, Kernel::ScalarBatch)?;

    let h32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let c32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let v32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaAlphaTrend::new(0).expect("CudaAlphaTrend::new");
    let batch = cuda
        .alphatrend_batch_dev(&h32, &l32, &c32, &v32, &sweep)
        .expect("alphatrend_batch_dev");

    assert_eq!(cpu.rows, batch.k1.rows);
    assert_eq!(cpu.cols, batch.k1.cols);
    assert_eq!(cpu.rows, batch.k2.rows);
    assert_eq!(cpu.cols, batch.k2.cols);

    let mut k1_host = vec![0f32; batch.k1.len()];
    let mut k2_host = vec![0f32; batch.k2.len()];
    batch.k1.buf.copy_to(&mut k1_host)?;
    batch.k2.buf.copy_to(&mut k2_host)?;

    let tol = 1e-3;
    for idx in 0..(cpu.rows * cpu.cols) {
        let ck1 = cpu.values_k1[idx];
        let gk1 = k1_host[idx] as f64;
        let ck2 = cpu.values_k2[idx];
        let gk2 = k2_host[idx] as f64;
        assert!(approx_eq(ck1, gk1, tol), "k1 mismatch at {}: cpu={} gpu={}", idx, ck1, gk1);
        assert!(approx_eq(ck2, gk2, tol), "k2 mismatch at {}: cpu={} gpu={}", idx, ck2, gk2);
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn alphatrend_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    use my_project::indicators::alphatrend::{alphatrend_into_slices, AlphaTrendInput};
    if !cuda_available() {
        eprintln!("[alphatrend_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 2048usize; // time
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    let mut vol_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s + 3)..rows { // stagger valid starts
            let idx = t * cols + s;
            let x = (t as f64) + (s as f64) * 0.3;
            high_tm[idx] = (x * 0.0009).sin() + 0.03;
            low_tm[idx] = high_tm[idx] - 0.02 - 0.005 * (x * 0.0017).cos().abs();
            close_tm[idx] = 0.5 * (high_tm[idx] + low_tm[idx]) + 0.0007 * (x * 0.0011).cos();
            vol_tm[idx] = (x * 0.0013).cos().abs() + 0.4;
        }
    }
    let coeff = 1.0f64;
    let period = 14usize;
    let no_volume = true;

    // CPU baseline per series
    let mut k1_cpu_tm = vec![f64::NAN; cols * rows];
    let mut k2_cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        let mut o = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = high_tm[idx];
            l[t] = low_tm[idx];
            c[t] = close_tm[idx];
            o[t] = c[t];
            v[t] = vol_tm[idx];
        }
        let input = AlphaTrendInput::from_slices(
            &o, &h, &l, &c, &v,
            AlphaTrendParams { coeff: Some(coeff), period: Some(period), no_volume: Some(no_volume) },
        );
        let mut k1 = vec![0.0; rows];
        let mut k2 = vec![0.0; rows];
        alphatrend_into_slices(&mut k1, &mut k2, &input, Kernel::Scalar)?;
        for t in 0..rows {
            k1_cpu_tm[t * cols + s] = k1[t];
            k2_cpu_tm[t * cols + s] = k2[t];
        }
    }

    // GPU
    let hf: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let lf: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let cf: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let vf: Vec<f32> = vol_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaAlphaTrend::new(0).expect("cuda alphatrend");
    let (k1_dev, k2_dev) = cuda
        .alphatrend_many_series_one_param_time_major_dev(
            &hf, &lf, &cf, &vf, cols, rows, coeff, period, no_volume,
        )
        .expect("alphatrend many-series");
    assert_eq!(k1_dev.rows, rows);
    assert_eq!(k1_dev.cols, cols);
    assert_eq!(k2_dev.rows, rows);
    assert_eq!(k2_dev.cols, cols);
    let mut k1_gpu_tm = vec![0f32; cols * rows];
    let mut k2_gpu_tm = vec![0f32; cols * rows];
    k1_dev.buf.copy_to(&mut k1_gpu_tm)?;
    k2_dev.buf.copy_to(&mut k2_gpu_tm)?;

    // Compare
    let tol = 2e-3;
    for idx in 0..(cols * rows) {
        let c1 = k1_cpu_tm[idx];
        let g1 = k1_gpu_tm[idx] as f64;
        let c2 = k2_cpu_tm[idx];
        let g2 = k2_gpu_tm[idx] as f64;
        assert!(approx_eq(c1, g1, tol), "k1 mismatch at {}", idx);
        assert!(approx_eq(c2, g2, tol), "k2 mismatch at {}", idx);
    }
    Ok(())
}
