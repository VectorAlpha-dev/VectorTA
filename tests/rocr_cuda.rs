// Integration tests for CUDA ROCR kernels

use my_project::indicators::rocr::{
    rocr_batch_with_kernel, rocr_with_kernel, RocrBatchRange, RocrInput, RocrParams, RocrData,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaRocr;

fn approx_eq(a: f64, b: f64, rel_tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    let diff = (a - b).abs();
    let scale = 1.0f64.max(a.abs()).max(b.abs());
    diff <= rel_tol * scale
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
fn rocr_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rocr_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 20000usize;
    let mut price = vec![f64::NAN; len];
    for i in 12..len { let x = i as f64; price[i] = (x*0.00123).sin() + 0.00011*x; }
    let sweep = RocrBatchRange { period: (5, 95, 5) };

    let cpu = rocr_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaRocr::new(0).expect("CudaRocr::new");
    let dev = cuda.rocr_batch_dev(&price_f32, &sweep).expect("rocr_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 2e-5; // relative tolerance
    for idx in 0..host.len() {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
        assert!(approx_eq(c, g, tol), "mismatch at {}: cpu={} gpu={}", idx, c, g);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn rocr_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rocr_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 4096usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols { for t in s..rows { let x=(t as f64)+(s as f64)*0.2; tm[t*cols+s]=(x*0.002).sin()+0.0003*x; } }
    let period = 14usize;

    // CPU per-series baseline
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = tm[t*cols + s]; }
        let input = RocrInput { data: RocrData::Slice(&series), params: RocrParams { period: Some(period) } };
        let out = rocr_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows { cpu_tm[t*cols + s] = out[t]; }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaRocr::new(0).expect("CudaRocr::new");
    let dev_tm = cuda
        .rocr_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period)
        .expect("rocr_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut host_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut host_tm)?;
    let tol = 2e-5; // relative tolerance
    for idx in 0..host_tm.len() {
        assert!(approx_eq(cpu_tm[idx], host_tm[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}
