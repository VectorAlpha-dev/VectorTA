// Integration tests for CUDA VIDYA kernels

use my_project::indicators::vidya::{
    vidya_batch_with_kernel, vidya_with_kernel, VidyaBatchRange, VidyaInput, VidyaParams, VidyaData,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaVidya;

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
fn vidya_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vidya_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut data = vec![f64::NAN; len];
    for i in 10..len {
        let x = i as f64;
        data[i] = (x * 0.00123).sin() + 0.00011 * x;
    }
    let sweep = VidyaBatchRange {
        short_period: (2, 8, 2),
        long_period: (10, 32, 11),
        alpha: (0.2, 0.2, 0.0),
    };

    let cpu = vidya_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaVidya::new(0).expect("CudaVidya::new");
    let dev = cuda
        .vidya_batch_dev(&data_f32, &sweep)
        .expect("vidya_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-4;
    for idx in 0..host.len() {
        assert!(approx_eq(cpu.values[idx], host[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn vidya_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vidya_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 1024usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.25;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let sp = 6usize;
    let lp = 21usize;
    let alpha = 0.2;

    // CPU per-series baseline
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = tm[t * cols + s]; }
        let input = VidyaInput { data: VidyaData::Slice(&series), params: VidyaParams{ short_period: Some(sp), long_period: Some(lp), alpha: Some(alpha) } };
        let out = vidya_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows { cpu_tm[t * cols + s] = out.values[t]; }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaVidya::new(0).expect("CudaVidya::new");
    let dev_tm = cuda
        .vidya_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &VidyaParams { short_period: Some(sp), long_period: Some(lp), alpha: Some(alpha) })
        .expect("vidya_many_series_one_param_time_major_dev");
    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut host_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut host_tm)?;

    let tol = 1e-4;
    for idx in 0..host_tm.len() {
        assert!(approx_eq(cpu_tm[idx], host_tm[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}

