// Integration tests for CUDA QStick kernels

use my_project::indicators::qstick::{
    qstick_batch_with_kernel, qstick_with_kernel, QstickBatchRange, QstickData, QstickInput,
    QstickParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaQstick;

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
fn qstick_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[qstick_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16_384usize;
    let mut open = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    for i in 3..len {
        let x = i as f64;
        open[i] = (x * 0.001).cos() + 0.0003 * x;
        close[i] = open[i] + 0.1 * (x * 0.0023).sin();
    }
    let sweep = QstickBatchRange {
        period: (5, 200, 5),
    };
    let cpu = qstick_batch_with_kernel(&open, &close, &sweep, Kernel::ScalarBatch)?;

    let open_f32: Vec<f32> = open.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let cuda = CudaQstick::new(0).expect("CudaQstick::new");
    let dev = cuda
        .qstick_batch_dev(&open_f32, &close_f32, &sweep)
        .expect("qstick_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
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
fn qstick_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[qstick_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize; // series
    let rows = 4096usize; // time
    let mut open_tm = vec![f64::NAN; cols * rows];
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let idx = t * cols + s;
            let x = (t as f64) + (s as f64) * 0.1;
            open_tm[idx] = (x * 0.002).cos() + 0.0002 * x;
            close_tm[idx] = open_tm[idx] + 0.08 * (x * 0.0031).sin();
        }
    }
    let period = 21usize;

    // CPU reference per series (column)
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut o = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            let idx = t * cols + s;
            o[t] = open_tm[idx];
            c[t] = close_tm[idx];
        }
        let params = QstickParams {
            period: Some(period),
        };
        let input = QstickInput {
            data: QstickData::Slices {
                open: &o,
                close: &c,
            },
            params,
        };
        let out = qstick_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let open_tm_f32: Vec<f32> = open_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaQstick::new(0).expect("CudaQstick::new");
    let dev_tm = cuda
        .qstick_many_series_one_param_time_major_dev(
            &open_tm_f32,
            &close_tm_f32,
            cols,
            rows,
            period,
        )
        .expect("qstick_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut host_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut host_tm)?;

    let tol = 1e-4;
    for idx in 0..host_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], host_tm[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
