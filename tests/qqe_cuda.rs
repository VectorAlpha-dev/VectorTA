// Integration tests for CUDA QQE kernels

use my_project::indicators::qqe::{
    qqe_batch_with_kernel, qqe_with_kernel, QqeBatchRange, QqeData, QqeInput, QqeParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::qqe_wrapper::CudaQqe;

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
fn qqe_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[qqe_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 64..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = QqeBatchRange {
        rsi_period: (8, 32, 4),
        smoothing_factor: (5, 9, 2),
        fast_factor: (4.236, 4.236, 0.0),
    };

    let cpu = qqe_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaQqe::new(0).expect("CudaQqe::new");
    let (dev, combos) = cuda
        .qqe_batch_dev(&price_f32, &sweep)
        .expect("qqe_batch_dev");

    assert_eq!(dev.cols, cpu.cols);
    assert_eq!(dev.rows, 2 * combos.len());
    assert_eq!(cpu.rows, combos.len());

    let mut out = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut out)?;

    let tol = 5e-4;
    let cols = cpu.cols;
    for row in 0..cpu.rows {
        let g_row_fast = 2 * row;
        let g_row_slow = 2 * row + 1;
        for t in 0..cols {
            let c_fast = cpu.fast_values[row * cols + t];
            let c_slow = cpu.slow_values[row * cols + t];
            let g_fast = out[g_row_fast * cols + t] as f64;
            let g_slow = out[g_row_slow * cols + t] as f64;
            assert!(
                approx_eq(c_fast, g_fast, tol),
                "fast mismatch at (row={}, t={})",
                row,
                t
            );
            assert!(
                approx_eq(c_slow, g_slow, tol),
                "slow mismatch at (row={}, t={})",
                row,
                t
            );
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn qqe_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[qqe_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 1024usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.17;
            tm[t * cols + s] = (x * 0.002).sin() + 0.00031 * x;
        }
    }
    let params = QqeParams {
        rsi_period: Some(14),
        smoothing_factor: Some(5),
        fast_factor: Some(4.236),
    };

    // CPU baseline per series
    let mut cpu_fast_tm = vec![f64::NAN; cols * rows];
    let mut cpu_slow_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let input = QqeInput {
            data: QqeData::Slice(&series),
            params: params.clone(),
        };
        let out = qqe_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_fast_tm[t * cols + s] = out.fast[t];
            cpu_slow_tm[t * cols + s] = out.slow[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaQqe::new(0).expect("CudaQqe::new");
    let dev = cuda
        .qqe_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("qqe_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, 2 * cols);

    let mut out = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut out)?;
    let tol = 1e-4;
    for s in 0..cols {
        for t in 0..rows {
            let g_fast = out[t * (2 * cols) + s] as f64;
            let g_slow = out[t * (2 * cols) + (s + cols)] as f64;
            let c_fast = cpu_fast_tm[t * cols + s];
            let c_slow = cpu_slow_tm[t * cols + s];
            assert!(
                approx_eq(c_fast, g_fast, tol),
                "fast mismatch at (s={}, t={})",
                s,
                t
            );
            assert!(
                approx_eq(c_slow, g_slow, tol),
                "slow mismatch at (s={}, t={})",
                s,
                t
            );
        }
    }

    Ok(())
}
