// Integration tests for CUDA CCI kernels

use my_project::indicators::cci::{
    CciBatchBuilder, CciBatchRange, CciBuilder, CciInput, CciParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaCci;

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
fn cci_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cci_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        price[i] = (x * 0.002).sin() + 0.0007 * x;
    }

    let sweep = CciBatchRange { period: (9, 64, 5) };

    let cpu = CciBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .period_range(sweep.period.0, sweep.period.1, sweep.period.2)
        .apply_slice(&price)?;

    let cuda = CudaCci::new(0).expect("CudaCci::new");
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let dev = cuda
        .cci_batch_dev(&price_f32, &sweep)
        .expect("cci cuda batch");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 6.5e-4; // f32 vs f64 baseline (slightly relaxed for FP32 parity)
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.values[idx], host[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cci_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[cci_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series count
    let rows = 2048usize; // series length
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for r in s..rows {
            let x = r as f64 + 0.13 * s as f64;
            tm[r * cols + s] = (x * 0.0023).sin() + 0.0002 * x;
        }
    }
    let period = 14usize;

    // CPU per-series baseline
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut col = vec![f64::NAN; rows];
        for r in 0..rows {
            col[r] = tm[r * cols + s];
        }
        let input = CciInput::from_slice(
            &col,
            CciParams {
                period: Some(period),
            },
        );
        let out = my_project::indicators::cci::cci_with_kernel(&input, Kernel::Scalar)?.values;
        for r in 0..rows {
            cpu_tm[r * cols + s] = out[r];
        }
    }

    // GPU
    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaCci::new(0).expect("CudaCci::new");
    let dev = cuda
        .cci_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period)
        .expect("cci many-series");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 1.1e-3;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}
