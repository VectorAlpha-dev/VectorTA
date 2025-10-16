// Integration tests for CUDA KST kernels (batch and many-series)

use my_project::indicators::kst::{KstBatchBuilder, KstBatchRange, KstBuilder, KstInput, KstParams};
use my_project::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use my_project::cuda::cuda_available;
#[cfg(feature = "cuda")]
use my_project::cuda::oscillators::CudaKst;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() { return true; }
    (a - b).abs() <= tol
}

#[test]
fn cuda_feature_off_noop_kst() {
    #[cfg(not(feature = "cuda"))]
    { assert!(true); }
}

#[cfg(feature = "cuda")]
#[test]
fn kst_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kst_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 8192usize;
    let mut price = vec![f64::NAN; len];
    for i in 5..len {
        let x = i as f64;
        price[i] = (x * 0.00123).sin() + 0.00017 * x;
    }

    let sweep = KstBatchRange {
        sma_period1: (5, 7, 1),
        sma_period2: (5, 5, 0),
        sma_period3: (5, 5, 0),
        sma_period4: (10, 10, 0),
        roc_period1: (5, 5, 0),
        roc_period2: (7, 7, 0),
        roc_period3: (10, 10, 0),
        roc_period4: (15, 15, 0),
        signal_period: (5, 5, 0),
    };

    let cpu = KstBatchBuilder::new()
        .kernel(Kernel::ScalarBatch)
        .sma_period1_range(sweep.sma_period1.0, sweep.sma_period1.1, sweep.sma_period1.2)
        .sma_period2_range(sweep.sma_period2.0, sweep.sma_period2.1, sweep.sma_period2.2)
        .sma_period3_range(sweep.sma_period3.0, sweep.sma_period3.1, sweep.sma_period3.2)
        .sma_period4_range(sweep.sma_period4.0, sweep.sma_period4.1, sweep.sma_period4.2)
        .roc_period1_range(sweep.roc_period1.0, sweep.roc_period1.1, sweep.roc_period1.2)
        .roc_period2_range(sweep.roc_period2.0, sweep.roc_period2.1, sweep.roc_period2.2)
        .roc_period3_range(sweep.roc_period3.0, sweep.roc_period3.1, sweep.roc_period3.2)
        .roc_period4_range(sweep.roc_period4.0, sweep.roc_period4.1, sweep.roc_period4.2)
        .signal_period_range(sweep.signal_period.0, sweep.signal_period.1, sweep.signal_period.2)
        .apply_slice(&price)?;

    let price_f32: Vec<f32> = price.iter().map(|&v| v as f32).collect();
    let cuda = CudaKst::new(0).expect("CudaKst::new");
    let (pair, _combos) = cuda.kst_batch_dev(&price_f32, &sweep).expect("kst_batch_dev");

    assert_eq!(cpu.rows, pair.rows());
    assert_eq!(cpu.cols, pair.cols());

    let mut g_line = vec![0f32; pair.line.len()];
    let mut g_sig  = vec![0f32; pair.signal.len()];
    pair.line.buf.copy_to(&mut g_line)?;
    pair.signal.buf.copy_to(&mut g_sig)?;

    let tol = 7e-4;
    for idx in 0..(cpu.rows * cpu.cols) {
        let c_l = cpu.lines[idx];
        let c_s = cpu.signals[idx];
        let g_l = g_line[idx] as f64;
        let g_s = g_sig[idx] as f64;
        assert!(approx_eq(c_l, g_l, tol), "line mismatch at {}: cpu={} gpu={}", idx, c_l, g_l);
        assert!(approx_eq(c_s, g_s, tol), "signal mismatch at {}: cpu={} gpu={}", idx, c_s, g_s);
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn kst_cuda_many_series_one_param_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kst_cuda_many_series_one_param_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let cols = 8usize; // number of series
    let rows = 4096usize; // time length
    let mut price_tm = vec![f64::NAN; cols * rows];
    for s in 0..cols { for t in 5..rows {
        let x = (t as f64) + (s as f64) * 0.1;
        price_tm[t * cols + s] = (x * 0.0019).sin() + 0.00011 * x;
    }}

    let params = KstParams { sma_period1: Some(10), sma_period2: Some(10), sma_period3: Some(10), sma_period4: Some(15),
                             roc_period1: Some(10), roc_period2: Some(15), roc_period3: Some(20), roc_period4: Some(30),
                             signal_period: Some(9) };

    // CPU baseline per series
    let mut cpu_line_tm = vec![f64::NAN; cols * rows];
    let mut cpu_sig_tm  = vec![f64::NAN; cols * rows];
    for s in 0..cols {
        let mut series = vec![f64::NAN; rows];
        for t in 0..rows { series[t] = price_tm[t * cols + s]; }
        let input = KstInput::from_slice(&series, params);
        let out = KstBuilder::new().apply_slice(&series)?; // or kst(&input)?
        for t in 0..rows {
            cpu_line_tm[t * cols + s] = out.line[t];
            cpu_sig_tm[t * cols + s]  = out.signal[t];
        }
    }

    let price_tm_f32: Vec<f32> = price_tm.iter().map(|&v| v as f32).collect();
    let cuda = CudaKst::new(0).expect("CudaKst::new");
    let pair_tm = cuda
        .kst_many_series_one_param_time_major_dev(&price_tm_f32, cols, rows, &params)
        .expect("kst many-series");

    assert_eq!(pair_tm.rows(), rows);
    assert_eq!(pair_tm.cols(), cols);

    let mut g_line_tm = vec![0f32; pair_tm.line.len()];
    let mut g_sig_tm  = vec![0f32; pair_tm.signal.len()];
    pair_tm.line.buf.copy_to(&mut g_line_tm)?;
    pair_tm.signal.buf.copy_to(&mut g_sig_tm)?;

    let tol = 9e-4;
    for idx in 0..g_line_tm.len() {
        assert!(approx_eq(cpu_line_tm[idx], g_line_tm[idx] as f64, tol), "line mismatch at {}", idx);
        assert!(approx_eq(cpu_sig_tm[idx],  g_sig_tm[idx]  as f64, tol), "signal mismatch at {}", idx);
    }

    Ok(())
}

