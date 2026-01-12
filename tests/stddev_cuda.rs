use vector_ta::indicators::stddev::{
    stddev_batch_with_kernel, stddev_with_kernel, StdDevBatchRange, StdDevInput, StdDevParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::CudaStddev;

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
fn stddev_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stddev_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut x = vec![f64::NAN; len];
    for i in 6..len {
        let t = i as f64;
        x[i] = (t * 0.00123).sin() + 0.0005 * t;
    }

    let sweep = StdDevBatchRange {
        period: (6, 30, 3),
        nbdev: (1.0, 2.0, 0.5),
    };
    let cpu = stddev_batch_with_kernel(&x, &sweep, Kernel::ScalarBatch)?;

    let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let cuda = CudaStddev::new(0).expect("CudaStddev::new");
    let dev = cuda
        .stddev_batch_dev(&x_f32, &sweep)
        .expect("stddev_cuda_batch_dev")
        .0;

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 5e-4;
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
fn stddev_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[stddev_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.1;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }

    let period = 15usize;
    let nbdev = 2.0f64;
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut col = vec![f64::NAN; rows];
        for t in 0..rows {
            col[t] = tm[t * cols + s];
        }
        let params = StdDevParams {
            period: Some(period),
            nbdev: Some(nbdev),
        };
        let input = StdDevInput::from_slice(&col, params);
        let out = stddev_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaStddev::new(0).expect("CudaStddev::new");
    let dev_tm = cuda
        .stddev_many_series_one_param_time_major_dev(&tm_f32, cols, rows, period, nbdev as f32)
        .expect("stddev_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);
    let mut host = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut host)?;
    let tol = 1e-4;
    for idx in 0..host.len() {
        assert!(
            approx_eq(cpu_tm[idx], host[idx] as f64, tol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
