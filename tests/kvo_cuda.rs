// Integration tests for CUDA KVO kernels

use my_project::indicators::kvo::{kvo_with_kernel, KvoBatchBuilder, KvoInput, KvoParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaKvo;

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
fn kvo_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kvo_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 32_768usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 6..len {
        let x = i as f64;
        let base = (x * 0.00123).sin() + 0.00017 * x;
        let spread = 0.1 * (x * 0.00077).cos().abs();
        high[i] = base + spread;
        low[i] = base - spread;
        close[i] = base + 0.05 * (x * 0.00111).sin();
        volume[i] = ((x * 0.0031).cos().abs() + 1.0) * 500.0;
    }

    let sweep = my_project::indicators::kvo::KvoBatchRange { short_period: (2, 8, 2), long_period: (10, 18, 2) };
    // Match CPU sweep to the CUDA sweep so dimensions align
    let cpu = KvoBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .short_range(sweep.short_period.0, sweep.short_period.1, sweep.short_period.2)
        .long_range(sweep.long_period.0, sweep.long_period.1, sweep.long_period.2)
        .apply_slices(&high, &low, &close, &volume)?;

    let h_f32: Vec<f32> = high.iter().copied().map(|v| v as f32).collect();
    let l_f32: Vec<f32> = low.iter().copied().map(|v| v as f32).collect();
    let c_f32: Vec<f32> = close.iter().copied().map(|v| v as f32).collect();
    let v_f32: Vec<f32> = volume.iter().copied().map(|v| v as f32).collect();

    let cuda = CudaKvo::new(0).expect("CudaKvo::new");
    let (dev, combos) = cuda.kvo_batch_dev(&h_f32, &l_f32, &c_f32, &v_f32, &sweep).expect("kvo_batch_dev");
    assert_eq!(dev.rows, cpu.rows);
    assert_eq!(dev.cols, cpu.cols);
    assert_eq!(combos.len(), cpu.rows);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 2e-1; // f32 GPU vs f64 CPU (absolute tolerance for large magnitudes)
    for idx in 0..host.len() {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(approx_eq(c, g, tol), "mismatch at {}: cpu={} gpu={}", idx, c, g);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn kvo_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kvo_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 4096usize;
    let mut h_tm = vec![f64::NAN; cols * rows];
    let mut l_tm = vec![f64::NAN; cols * rows];
    let mut c_tm = vec![f64::NAN; cols * rows];
    let mut v_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in (s + 3)..rows {
            let x = t as f64 + s as f64 * 0.2;
            let base = (x * 0.0021).cos() + 0.0004 * x;
            let spread = 0.08 * (x * 0.001).sin().abs();
            let idx = t * cols + s;
            h_tm[idx] = base + spread;
            l_tm[idx] = base - spread;
            c_tm[idx] = base + 0.03 * (x * 0.0017).sin();
            v_tm[idx] = ((x * 0.0042).sin().abs() + 0.9) * 300.0;
        }
    }

    let short = 6usize;
    let long = 20usize;

    // CPU baseline per series
    let mut cpu = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            h[t] = h_tm[idx];
            l[t] = l_tm[idx];
            c[t] = c_tm[idx];
            v[t] = v_tm[idx];
        }
        let params = KvoParams { short_period: Some(short), long_period: Some(long) };
        let input = KvoInput::from_slices(&h, &l, &c, &v, params);
        let out = kvo_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu[t * cols + s] = out[t];
        }
    }

    let h_f32: Vec<f32> = h_tm.iter().map(|&v| v as f32).collect();
    let l_f32: Vec<f32> = l_tm.iter().map(|&v| v as f32).collect();
    let c_f32: Vec<f32> = c_tm.iter().map(|&v| v as f32).collect();
    let v_f32: Vec<f32> = v_tm.iter().map(|&v| v as f32).collect();

    let cuda = CudaKvo::new(0).expect("CudaKvo::new");
    let dev = cuda
        .kvo_many_series_one_param_time_major_dev(&h_f32, &l_f32, &c_f32, &v_f32, cols, rows, &KvoParams { short_period: Some(short), long_period: Some(long) })
        .expect("kvo many series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut g = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g)?;

    let tol = 2.5; // absolute tolerance; FP32/FP64 mixed path (GPU)
    for idx in 0..g.len() {
        assert!(approx_eq(cpu[idx], g[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}

