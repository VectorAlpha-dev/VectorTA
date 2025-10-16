// Integration tests for CUDA VOSC kernels

use my_project::indicators::vosc::{
    vosc_batch_with_kernel, vosc_with_kernel, VoscBatchRange, VoscData, VoscInput, VoscParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaVosc;

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
fn vosc_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vosc_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 32_768usize;
    let mut volume = vec![f64::NAN; len];
    for i in 7..len { let x = i as f64; volume[i] = (x * 0.0013).cos() + 0.02 * (x * 0.0021).sin(); }
    let sweep = VoscBatchRange { short_period: (2, 50, 3), long_period: (10, 120, 5) };

    let cpu = vosc_batch_with_kernel(&volume, &sweep, Kernel::ScalarBatch)?;

    let vol_f32: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
    let cuda = CudaVosc::new(0).expect("CudaVosc::new");
    let (dev, combos) = cuda
        .vosc_batch_dev(&vol_f32, &sweep)
        .expect("vosc_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(combos.len(), cpu.rows);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;
    let tol = 5e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
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
fn vosc_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vosc_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize; // series
    let rows = 8192usize; // time
    let mut vol_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols { for t in (s+3)..rows { let idx = t * cols + s; let x = (t as f64) + (s as f64) * 0.1; vol_tm[idx] = (x * 0.0017).cos() + 0.03 * (x * 0.0037).sin(); } }
    let params = VoscParams { short_period: Some(5), long_period: Some(34) };

    // CPU reference per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut v = vec![f64::NAN; rows];
        for t in 0..rows { let idx = t * cols + s; v[t] = (vol_tm[idx] as f32) as f64; }
        let input = VoscInput { data: VoscData::Slice(&v), params: params.clone() };
        let out = vosc_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows { cpu_tm[t * cols + s] = out.values[t]; }
    }

    let vol_tm_f32: Vec<f32> = vol_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVosc::new(0).expect("CudaVosc::new");
    let dev_tm = cuda
        .vosc_many_series_one_param_time_major_dev(&vol_tm_f32, cols, rows, &params)
        .expect("vosc_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut host_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut host_tm)?;

    let tol = 5e-3;
    for idx in 0..host_tm.len() {
        assert!(approx_eq(cpu_tm[idx], host_tm[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}
