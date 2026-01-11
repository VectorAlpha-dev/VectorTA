

use vector_ta::indicators::macd::{macd, MacdBatchBuilder, MacdBatchRange, MacdInput, MacdParams};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::cuda_available;
#[cfg(feature = "cuda")]
use vector_ta::cuda::oscillators::CudaMacd;

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
fn macd_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[macd_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }
    let sweep = MacdBatchRange {
        fast_period: (10, 18, 2),
        slow_period: (26, 26, 0),
        signal_period: (9, 9, 0),
        ma_type: ("ema".to_string(), "ema".to_string(), String::new()),
    };

    let cpu =
        vector_ta::indicators::macd::macd_batch_with_kernel(&price, &sweep, Kernel::ScalarBatch)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaMacd::new(0).expect("CudaMacd::new");
    let (dev_triplet, combos) = cuda
        .macd_batch_dev(&price_f32, &sweep)
        .expect("macd_batch_dev");
    assert_eq!(combos.len(), cpu.rows);
    assert_eq!(dev_triplet.macd.rows, cpu.rows);
    assert_eq!(dev_triplet.macd.cols, cpu.cols);

    let mut g_macd = vec![0f32; dev_triplet.macd.len()];
    let mut g_signal = vec![0f32; dev_triplet.signal.len()];
    let mut g_hist = vec![0f32; dev_triplet.hist.len()];
    dev_triplet.macd.buf.copy_to(&mut g_macd)?;
    dev_triplet.signal.buf.copy_to(&mut g_signal)?;
    dev_triplet.hist.buf.copy_to(&mut g_hist)?;

    let tol = 5e-4f64;
    for idx in 0..(cpu.rows * cpu.cols) {
        assert!(
            approx_eq(cpu.macd[idx], g_macd[idx] as f64, tol),
            "macd mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.signal[idx], g_signal[idx] as f64, tol),
            "signal mismatch at {}",
            idx
        );
        assert!(
            approx_eq(cpu.hist[idx], g_hist[idx] as f64, tol),
            "hist mismatch at {}",
            idx
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn macd_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[macd_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }
    let cols = 8usize;
    let rows = 2048usize;
    let mut tm = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        for t in s..rows {
            let x = (t as f64) + (s as f64) * 0.3;
            tm[t * cols + s] = (x * 0.002).sin() + 0.0002 * x;
        }
    }
    let fast = 12usize;
    let slow = 26usize;
    let signal = 9usize;

    
    let mut cpu_macd = vec![f64::NAN; cols * rows];
    let mut cpu_signal = vec![f64::NAN; cols * rows];
    let mut cpu_hist = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut p = vec![f64::NAN; rows];
        for t in 0..rows {
            p[t] = tm[t * cols + s];
        }
        let input = MacdInput::from_slice(
            &p,
            MacdParams {
                fast_period: Some(fast),
                slow_period: Some(slow),
                signal_period: Some(signal),
                ma_type: Some("ema".to_string()),
            },
        );
        let out = macd(&input)?;
        for t in 0..rows {
            cpu_macd[t * cols + s] = out.macd[t];
            cpu_signal[t * cols + s] = out.signal[t];
            cpu_hist[t * cols + s] = out.hist[t];
        }
    }

    let tm_f32: Vec<f32> = tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaMacd::new(0).expect("CudaMacd::new");
    let params = MacdParams {
        fast_period: Some(fast),
        slow_period: Some(slow),
        signal_period: Some(signal),
        ma_type: Some("ema".to_string()),
    };
    let dev = cuda
        .macd_many_series_one_param_time_major_dev(&tm_f32, cols, rows, &params)
        .expect("macd many series");
    assert_eq!(dev.macd.rows, rows);
    assert_eq!(dev.macd.cols, cols);
    let mut g_macd = vec![0f32; dev.macd.len()];
    let mut g_signal = vec![0f32; dev.signal.len()];
    let mut g_hist = vec![0f32; dev.hist.len()];
    dev.macd.buf.copy_to(&mut g_macd)?;
    dev.signal.buf.copy_to(&mut g_signal)?;
    dev.hist.buf.copy_to(&mut g_hist)?;

    let tol = 1e-4;
    for i in 0..(cols * rows) {
        assert!(
            approx_eq(cpu_macd[i], g_macd[i] as f64, tol),
            "macd mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_signal[i], g_signal[i] as f64, tol),
            "signal mismatch at {}",
            i
        );
        assert!(
            approx_eq(cpu_hist[i], g_hist[i] as f64, tol),
            "hist mismatch at {}",
            i
        );
    }
    Ok(())
}
