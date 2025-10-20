// Integration tests for CUDA EMV kernels

use my_project::indicators::emv::{emv_with_kernel, EmvInput};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaEmv;

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
fn emv_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[emv_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16384usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        close[i] = (x * 0.00123).sin() + 0.00031 * x;
        let off = (x * 0.0021).cos().abs() + 0.12;
        high[i] = close[i] + off;
        low[i] = close[i] - off;
        volume[i] = ((x * 0.0067).sin().abs() + 0.9) * 500.0 + 100.0;
    }

    // CPU baseline
    let cpu = {
        let input = EmvInput::from_slices(&high, &low, &close, &volume);
        emv_with_kernel(&input, Kernel::Scalar)?.values
    };

    // GPU
    let h32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let l32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let v32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaEmv::new(0).expect("CudaEmv::new");
    let dev = cuda.emv_batch_dev(&h32, &l32, &v32).expect("emv_batch_dev");

    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);
    let mut gpu = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu)?;

    let tol = 5e-4;
    for i in 0..len {
        assert!(
            approx_eq(cpu[i], gpu[i] as f64, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            cpu[i],
            gpu[i]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn emv_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[emv_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 4096usize;
    let mut high_tm = vec![f64::NAN; cols * rows];
    let mut low_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    let mut vol_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows {
            let idx = r * cols + s;
            let x = r as f64 + s as f64 * 0.2;
            close_tm[idx] = (x * 0.0027).cos() + 0.00029 * x;
            let off = (x * 0.0017).sin().abs() + 0.08;
            high_tm[idx] = close_tm[idx] + off;
            low_tm[idx] = close_tm[idx] - off;
            vol_tm[idx] = ((x * 0.0063).sin().abs() + 0.9) * 420.0 + 80.0;
        }
    }

    // CPU per-series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for r in 0..rows {
            let idx = r * cols + s;
            h[r] = high_tm[idx];
            l[r] = low_tm[idx];
            c[r] = close_tm[idx];
            v[r] = vol_tm[idx];
        }
        let input = EmvInput::from_slices(&h, &l, &c, &v);
        let out = emv_with_kernel(&input, Kernel::Scalar)?.values;
        for r in 0..rows {
            cpu_tm[r * cols + s] = out[r];
        }
    }

    // GPU
    let high_tm32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let vol_tm32: Vec<f32> = vol_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaEmv::new(0).expect("CudaEmv::new");
    let dev_tm = cuda
        .emv_many_series_one_param_time_major_dev(&high_tm32, &low_tm32, &vol_tm32, cols, rows)
        .expect("emv_many_series_one_param_time_major_dev");
    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut gpu_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut gpu_tm)?;

    let tol = 1e-3;
    for idx in 0..gpu_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
