// Integration tests for CUDA Reverse RSI kernels

use my_project::indicators::reverse_rsi::{
    reverse_rsi_batch_with_kernel, reverse_rsi_with_kernel, ReverseRsiBatchRange, ReverseRsiInput,
    ReverseRsiParams,
};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaReverseRsi;

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
fn reverse_rsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[reverse_rsi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 16384usize;
    let mut price = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00011 * x;
    }
    let sweep = ReverseRsiBatchRange {
        rsi_length_range: (7, 21, 7),
        rsi_level_range: (30.0, 70.0, 20.0),
    };

    // Align CPU baseline to FP32 input used on GPU
    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let price_cpu_f64: Vec<f64> = price_f32.iter().map(|&v| v as f64).collect();
    let cpu = reverse_rsi_batch_with_kernel(&price_cpu_f64, &sweep, Kernel::ScalarBatch)?;

    let cuda = CudaReverseRsi::new(0).expect("CudaReverseRsi::new");
    let (dev, _combos) = cuda
        .reverse_rsi_batch_dev(&price_f32, &sweep)
        .expect("reverse_rsi_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let tol = 7e-4; // conservative due to FP32 accumulation
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
fn reverse_rsi_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[reverse_rsi_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize;
    let rows = 4096usize;
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            price_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }

    let rsi_length = 14usize;
    let rsi_level = 55.0f64;

    // CPU baseline per series
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = price_tm[t * cols + s];
        }
        // Align CPU baseline to FP32 input used on GPU
        let p_f32: Vec<f32> = p.iter().map(|&v| v as f32).collect();
        let p_cpu_f64: Vec<f64> = p_f32.iter().map(|&v| v as f64).collect();
        let params = ReverseRsiParams {
            rsi_length: Some(rsi_length),
            rsi_level: Some(rsi_level),
        };
        let input = ReverseRsiInput::from_slice(&p_cpu_f64, params);
        let out = reverse_rsi_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaReverseRsi::new(0).expect("CudaReverseRsi::new");
    let dev_tm = cuda
        .reverse_rsi_many_series_one_param_time_major_dev(
            &price_tm_f32,
            cols,
            rows,
            &ReverseRsiParams {
                rsi_length: Some(rsi_length),
                rsi_level: Some(rsi_level),
            },
        )
        .expect("reverse_rsi_many_series_one_param_time_major_dev");

    assert_eq!(dev_tm.rows, rows);
    assert_eq!(dev_tm.cols, cols);

    let mut g_tm = vec![0f32; dev_tm.len()];
    dev_tm.buf.copy_to(&mut g_tm)?;

    let tol = 7e-4;
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, tol),
            "many-series mismatch at {}",
            idx
        );
    }

    Ok(())
}
