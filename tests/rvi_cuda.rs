// Integration tests for CUDA RVI kernels

use my_project::indicators::rvi::{
    rvi_batch_with_kernel, rvi_with_kernel, RviBatchRange, RviInput, RviParams,
};
use my_project::indicators::rvi::{
    rvi_batch_with_kernel, rvi_with_kernel, RviBatchRange, RviInput, RviParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaRvi;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
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
    {
        assert!(true);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn rvi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rvi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    if !cuda_available() {
        eprintln!("[rvi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 131072usize;
    let mut data = vec![f64::NAN; len];
    for i in 200..len {
        let x = i as f64;
        data[i] = (x * 0.0011).sin() + 0.00021 * x;
    }
    for i in 200..len {
        let x = i as f64;
        data[i] = (x * 0.0011).sin() + 0.00021 * x;
    }
    // Keep to devtype=0 (StdDev) and matype=1 (EMA) for CUDA parity
    let sweep = RviBatchRange {
        period: (10, 24, 2),
        ma_len: (14, 14, 0),
        matype: (1, 1, 0),
        devtype: (0, 0, 0),
    };
    let sweep = RviBatchRange {
        period: (10, 24, 2),
        ma_len: (14, 14, 0),
        matype: (1, 1, 0),
        devtype: (0, 0, 0),
    };

    let cpu = rvi_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;

    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaRvi::new(0).expect("CudaRvi::new");
    let (dev, combos) = cuda
        .rvi_batch_dev(&data_f32, &sweep)
        .expect("rvi_batch_dev");
    let (dev, combos) = cuda
        .rvi_batch_dev(&data_f32, &sweep)
        .expect("rvi_batch_dev");
    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);
    assert_eq!(cpu.rows, combos.len());

    let mut gpu = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu)?;

    let tol = 15.0; // tolerance for FP32 vs FP64
    for idx in 0..(cpu.rows * cpu.cols) {
        if !approx_eq(cpu.values[idx], gpu[idx] as f64, tol) {
            eprintln!("first mismatch at {}: cpu={} gpu={}", idx, cpu.values[idx], gpu[idx]);
            assert!(false, "mismatch at {}", idx);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn rvi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rvi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    if !cuda_available() {
        eprintln!("[rvi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 16usize; // series
    let rows = 4096usize; // time
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = t as f64 + (s as f64) * 0.25;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let params = RviParams {
        period: Some(10),
        ma_len: Some(14),
        matype: Some(1),
        devtype: Some(0),
    };
    for s in 0..cols {
        for t in s..rows {
            let x = t as f64 + (s as f64) * 0.25;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let params = RviParams {
        period: Some(10),
        ma_len: Some(14),
        matype: Some(1),
        devtype: Some(0),
    };

    // CPU baseline per series (scalar)
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        for t in 0..rows {
            series[t] = tm[t * cols + s];
        }
        let input = RviInput::from_slice(&series, params.clone());
        let out = rvi_with_kernel(&input, Kernel::Scalar)?.values;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
        for t in 0..rows {
            cpu_tm[t * cols + s] = out[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaRvi::new(0).expect("CudaRvi::new");
    let dev = cuda
        .rvi_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("rvi_many_series_one_param_time_major_dev");
    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);

    let mut gpu_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut gpu_tm)?;

    let tol = 6.0; // tolerance for FP32 vs FP64
    for idx in 0..gpu_tm.len() {
        if !approx_eq(cpu_tm[idx], gpu_tm[idx] as f64, tol) {
            eprintln!("first mismatch at {}: cpu={} gpu={}", idx, cpu_tm[idx], gpu_tm[idx]);
            assert!(false, "mismatch at {}", idx);
        }
    }
    Ok(())
}
