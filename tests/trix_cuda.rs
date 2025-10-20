// Integration tests for CUDA TRIX kernels

use my_project::indicators::trix::{
    trix_batch_with_kernel, TrixBatchRange, TrixData, TrixInput, TrixParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::moving_averages::CudaTrix;

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
fn trix_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trix_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        // Strictly positive series for ln()
        price[i] = 2.0 + 0.01 * x + (x * 0.00123).sin().abs();
    }
    // CPU baseline on FP32-rounded inputs to match CUDA math
    let price32_as_f64: Vec<f64> = price.iter().map(|&v| (v as f32) as f64).collect();
    let sweep = TrixBatchRange { period: (8, 64, 7) };
    let cpu = trix_batch_with_kernel(&price32_as_f64, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaTrix::new(0).expect("CudaTrix::new");
    let dev = cuda
        .trix_batch_dev(&price_f32, &sweep)
        .expect("trix_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-3; // TRIX scaled by 1e4; allow small fp32 drift
    for idx in 0..(cpu.rows * cpu.cols) {
        let c = cpu.values[idx];
        let g = host[idx] as f64;
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

#[cfg(feature = "cuda")]
#[test]
fn trix_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trix_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 2048usize; // length
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in 3..rows {
            let x = (t as f64) * 0.002 + (s as f64) * 0.05;
            price_tm[t * cols + s] = 3.0 + 0.003 * (t as f64) + (x * 0.7).sin().abs();
        }
    }

    let period = 18usize;

    // CPU baseline per series (computed on FP32-rounded data to match GPU inputs)
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = (price_tm[t * cols + s] as f32) as f64;
        }
        let input = TrixInput {
            data: TrixData::Slice(&p),
            params: TrixParams {
                period: Some(period),
            },
        };
        let out = my_project::indicators::trix::trix(&input)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaTrix::new(0).expect("CudaTrix::new");
    let dev_tm = cuda
        .trix_many_series_one_param_time_major_dev(&price_tm_f32, cols, rows, period)
        .expect("trix many-series tm");

    assert_eq!(dev_tm.cols, cols);
    assert_eq!(dev_tm.rows, rows);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 5e-3;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }

    Ok(())
}
