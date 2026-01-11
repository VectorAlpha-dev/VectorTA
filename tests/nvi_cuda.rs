

use vector_ta::indicators::nvi::{nvi_with_kernel, NviInput, NviParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaNvi;

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
fn nvi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[nvi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    
    for i in 3..len {
        let x = i as f64;
        close[i] = (x * 0.00123).sin() + 100.0 + 0.00017 * x;
        volume[i] = (x * 0.00077).cos().abs() * 500.0 + 100.0;
    }
    let input = NviInput::from_slices(&close, &volume, NviParams);
    let cpu = nvi_with_kernel(&input, Kernel::Scalar)?.values;

    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaNvi::new(0).expect("CudaNvi::new");
    let dev = cuda
        .nvi_batch_dev(&close_f32, &volume_f32)
        .expect("nvi_batch_dev");
    assert_eq!(dev.rows, 1);
    assert_eq!(dev.cols, len);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 2e-4;
    for i in 0..len {
        let g = host[i] as f64;
        let c = cpu[i];
        assert!(
            approx_eq(c, g, tol),
            "mismatch at {}: cpu={} gpu={}",
            i,
            c,
            g
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn nvi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[nvi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize;
    let rows = 2048usize;
    let mut close_tm = vec![f64::NAN; cols * rows];
    let mut volume_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s.min(5)..rows {
            let x = (t as f64) + (s as f64) * 0.25;
            close_tm[t * cols + s] = (x * 0.0017).sin() + 100.0 + 0.0003 * x;
            volume_tm[t * cols + s] = (x * 0.0011).cos().abs() * 400.0 + 120.0;
        }
    }

    
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut c = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows {
            c[t] = close_tm[t * cols + s];
            v[t] = volume_tm[t * cols + s];
        }
        let input = NviInput::from_slices(&c, &v, NviParams);
        let out = nvi_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let close_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let volume_f32: Vec<f32> = volume_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaNvi::new(0).expect("CudaNvi::new");
    let dev = cuda
        .nvi_many_series_one_param_time_major_dev(&close_f32, &volume_f32, cols, rows)
        .expect("nvi_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 2e-4;
    for idx in 0..host.len() {
        let g = host[idx] as f64;
        let c = cpu_tm[idx];
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
