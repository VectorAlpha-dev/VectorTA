// CUDA integration tests for Chande (Chandelier Exit)

use my_project::indicators::chande::{
    chande_batch_with_kernel, chande_with_kernel, ChandeBatchRange, ChandeInput, ChandeParams,
};
use my_project::utilities::data_loader::{read_candles_from_csv, Candles};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::CudaChande;

fn approx_eq(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let diff = (a - b).abs();
    diff <= atol + rtol * a.abs().max(b.abs())
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
fn chande_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[chande_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    // Use dataset from repo to ensure realistic HLC
    let file = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv";
    let candles: Candles = read_candles_from_csv(file)?;
    let high: Vec<f64> = candles.high.clone();
    let low: Vec<f64> = candles.low.clone();
    let close: Vec<f64> = candles.close.clone();

    let sweep = ChandeBatchRange {
        period: (10, 30, 5),
        mult: (2.0, 4.0, 1.0),
    };
    let dir = "long";

    // Quantize CPU baseline to f32 to align with GPU FP32 inputs
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let close_f32: Vec<f32> = close.iter().map(|&v| v as f32).collect();
    let high_q: Vec<f64> = high_f32.iter().map(|&v| v as f64).collect();
    let low_q: Vec<f64> = low_f32.iter().map(|&v| v as f64).collect();
    let close_q: Vec<f64> = close_f32.iter().map(|&v| v as f64).collect();
    let cpu =
        chande_batch_with_kernel(&high_q, &low_q, &close_q, &sweep, dir, Kernel::ScalarBatch)?;

    // Reuse quantized f32 inputs
    let cuda = CudaChande::new(0).expect("CudaChande::new");
    let dev = cuda
        .chande_batch_dev(&high_f32, &low_f32, &close_f32, &sweep, dir)
        .expect("chande_batch_dev");

    assert_eq!(cpu.rows, dev.rows);
    assert_eq!(cpu.cols, dev.cols);

    let mut host = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut host)?;

    let (atol, rtol) = (1e-3, 1e-6);
    for idx in 0..(cpu.rows * cpu.cols) {
        // Quantize CPU output to FP32 to match GPU result scale
        let c = (cpu.values[idx] as f32) as f64;
        let g = host[idx] as f64;
        assert!(
            approx_eq(c, g, atol, rtol),
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
fn chande_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[chande_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // series
    let rows = 4096usize; // time
                          // Synthesize HLC time-major arrays
    let mut close_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.2;
            close_tm[t * cols + s] = (x * 0.002).sin() + 0.0003 * x;
        }
    }
    let (mut high_tm, mut low_tm) = (close_tm.clone(), close_tm.clone());
    for s in 0..cols {
        for t in 0..rows {
            let v = close_tm[t * cols + s];
            if v.is_nan() {
                continue;
            }
            let x = (t as f64) * 0.002;
            let off = (0.004 * x.cos()).abs() + 0.11;
            high_tm[t * cols + s] = v + off;
            low_tm[t * cols + s] = v - off;
        }
    }

    // CPU baseline per series using scalar kernel
    let period = 22usize;
    let mult = 3.0f64;
    let direction = "long";
    let mut cpu_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut h = vec![f64::NAN; rows];
        let mut l = vec![f64::NAN; rows];
        let mut c = vec![f64::NAN; rows];
        for t in 0..rows {
            h[t] = high_tm[t * cols + s];
            l[t] = low_tm[t * cols + s];
            c[t] = close_tm[t * cols + s];
        }
        let params = ChandeParams {
            period: Some(period),
            mult: Some(mult),
            direction: Some(direction.into()),
        };
        let input = ChandeInput::from_slices(&h, &l, &c, params);
        let out = chande_with_kernel(&input, Kernel::Scalar)?;
        for t in 0..rows {
            cpu_tm[t * cols + s] = out.values[t];
        }
    }

    let high_tm_f32: Vec<f32> = high_tm.iter().map(|&v| v as f32).collect();
    let low_tm_f32: Vec<f32> = low_tm.iter().map(|&v| v as f32).collect();
    let close_tm_f32: Vec<f32> = close_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaChande::new(0).expect("CudaChande::new");
    let dev = cuda
        .chande_many_series_one_param_time_major_dev(
            &high_tm_f32,
            &low_tm_f32,
            &close_tm_f32,
            cols,
            rows,
            period,
            mult as f32,
            direction,
        )
        .expect("chande_many_series_one_param_time_major_dev");

    assert_eq!(dev.rows, rows);
    assert_eq!(dev.cols, cols);
    let mut g_tm = vec![0f32; dev.len()];
    dev.buf.copy_to(&mut g_tm)?;

    let (atol, rtol) = (1e-3, 1e-6);
    for idx in 0..g_tm.len() {
        assert!(
            approx_eq(cpu_tm[idx], g_tm[idx] as f64, atol, rtol),
            "mismatch at {}",
            idx
        );
    }
    Ok(())
}
