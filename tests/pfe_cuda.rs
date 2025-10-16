// CUDA integration tests for Polarized Fractal Efficiency (PFE)

use my_project::utilities::enums::Kernel;
use my_project::indicators::pfe::{
    pfe_batch_with_kernel, pfe_with_kernel, PfeBatchRange, PfeInput, PfeParams, PfeData,
};

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::pfe_wrapper::CudaPfe;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { true } else { (a - b).abs() <= tol }
}

#[test]
fn cuda_feature_off_noop() { #[cfg(not(feature = "cuda"))] assert!(true); }

#[cfg(feature = "cuda")]
#[test]
fn pfe_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pfe_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![0.0f64; len];
    for i in 0..len { let x = i as f64; price[i] = (x * 0.00123).sin() + 0.00017 * x; }
    let sweep = PfeBatchRange { period: (5, 45, 5), smoothing: (3, 9, 2) };

    // CPU baseline
    let cpu = pfe_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    // GPU
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaPfe::new(0).expect("CudaPfe::new");
    let dev = cuda.pfe_batch_dev(&price_f32, &sweep).expect("pfe_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    let mut got = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got)?;

    let tol = 5e-2; // FP32 drift tolerance (EMA + sqrt accumulations)
    for idx in 0..got.len() {
        if !approx_eq(cpu.values[idx], got[idx] as f64, tol) {
            if std::env::var("PFE_DEBUG").ok().as_deref() == Some("1") {
                eprintln!("idx={} cpu={} gpu={}", idx, cpu.values[idx], got[idx]);
                // Dump a small window
                let row0 = 0usize;
                let cols = cpu.cols;
                for k in 0..6 {
                    let t = (idx % cols) + k;
                    if t < cols {
                        eprintln!(
                            "t={} cpu={} gpu={}",
                            t,
                            cpu.values[row0 * cols + t],
                            got[row0 * cols + t]
                        );
                    }
                }
            }
            assert!(false, "mismatch at {}", idx);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn pfe_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pfe_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 1024usize; // time
    let mut tm = vec![0.0f64; cols * rows];
    for s in 0..cols { for t in 0..rows { let x = (t as f64) + (s as f64) * 0.2; tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x; } }
    let period = 10usize; let smoothing = 5usize;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = tm[t * cols + s]; }
        let params = PfeParams { period: Some(period), smoothing: Some(smoothing) };
        let input = PfeInput { data: PfeData::Slice(&series), params };
        let out = pfe_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows { cpu_tm[t * cols + s] = out.values[t]; }
    }

    // GPU
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaPfe::new(0).expect("CudaPfe::new");
    let dev = cuda
        .pfe_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period, smoothing)
        .expect("pfe_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut got_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut got_tm)?;

    let tol = 5e-2;
    for idx in 0..got_tm.len() {
        assert!(approx_eq(cpu_tm[idx], got_tm[idx] as f64, tol), "mismatch at {}", idx);
    }
    Ok(())
}
