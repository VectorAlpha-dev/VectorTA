use vector_ta::indicators::vwap_deviation_oscillator::{
    vwap_deviation_oscillator_batch_with_kernel, VwapDeviationMode,
    VwapDeviationOscillatorBatchRange, VwapDeviationSessionMode,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVwapDeviationOscillator};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlcv(len: usize) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut timestamps = vec![0_i64; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let start = 1_700_000_000_000_i64;
    let step = 6_i64 * 60 * 60 * 1_000;
    let mut base = 103.0f64;

    for i in 0..len {
        timestamps[i] = start + (i as i64) * step;
    }

    for i in 18..len {
        let x = i as f64;
        base += (x * 0.007).sin() * 0.34 + (x * 0.0025).cos() * 0.11;
        let c = base + (x * 0.019).sin() * 1.12 + (x * 0.014).cos() * 0.37;
        let span = 0.92 + (x * 0.013).sin().abs() * 0.46;
        close[i] = c;
        high[i] = c + span;
        low[i] = c - span * (0.78 + (x * 0.011).cos().abs() * 0.22);
        volume[i] = 32_000.0 + (x * 0.021).sin() * 3_800.0 + (x % 17.0) * 113.0;
    }

    for gap in [211usize, 377usize] {
        high[gap] = f64::NAN;
        low[gap] = f64::NAN;
        close[gap] = f64::NAN;
        volume[gap] = f64::NAN;
    }

    (timestamps, high, low, close, volume)
}

#[cfg(feature = "cuda")]
fn run_case(sweep: VwapDeviationOscillatorBatchRange) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vwap_deviation_oscillator_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (timestamps, high, low, close, volume) = sample_ohlcv(640);
    let cpu = vwap_deviation_oscillator_batch_with_kernel(
        &timestamps,
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaVwapDeviationOscillator::new(0)?;
    let result = cuda.batch_dev(&timestamps, &high, &low, &close, &volume, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_osc = vec![0.0f64; result.outputs.osc.len()];
    let mut got_std1 = vec![0.0f64; result.outputs.std1.len()];
    let mut got_std2 = vec![0.0f64; result.outputs.std2.len()];
    let mut got_std3 = vec![0.0f64; result.outputs.std3.len()];
    result.outputs.osc.buf.copy_to(&mut got_osc)?;
    result.outputs.std1.buf.copy_to(&mut got_std1)?;
    result.outputs.std2.buf.copy_to(&mut got_std2)?;
    result.outputs.std3.buf.copy_to(&mut got_std3)?;

    for idx in 0..cpu.osc.len() {
        assert!(approx_eq(cpu.osc[idx], got_osc[idx], 1e-9));
        assert!(approx_eq(cpu.std1[idx], got_std1[idx], 1e-9));
        assert!(approx_eq(cpu.std2[idx], got_std2[idx], 1e-9));
        assert!(approx_eq(cpu.std3[idx], got_std3[idx], 1e-9));
    }

    Ok(())
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn vwap_deviation_oscillator_cuda_absolute_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(VwapDeviationOscillatorBatchRange {
        rolling_period: (18, 22, 4),
        rolling_days: (30, 30, 0),
        z_window: (50, 50, 0),
        pct_vol_lookback: (100, 100, 0),
        pct_min_sigma: (0.1, 0.1, 0.0),
        abs_vol_lookback: (24, 28, 4),
        session_mode: VwapDeviationSessionMode::RollingBars,
        use_close: false,
        deviation_mode: VwapDeviationMode::Absolute,
    })
}

#[cfg(feature = "cuda")]
#[test]
fn vwap_deviation_oscillator_cuda_percent_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(VwapDeviationOscillatorBatchRange {
        rolling_period: (20, 20, 0),
        rolling_days: (30, 30, 0),
        z_window: (50, 50, 0),
        pct_vol_lookback: (80, 120, 40),
        pct_min_sigma: (0.1, 0.2, 0.1),
        abs_vol_lookback: (100, 100, 0),
        session_mode: VwapDeviationSessionMode::Daily,
        use_close: false,
        deviation_mode: VwapDeviationMode::Percent,
    })
}

#[cfg(feature = "cuda")]
#[test]
fn vwap_deviation_oscillator_cuda_zscore_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(VwapDeviationOscillatorBatchRange {
        rolling_period: (20, 20, 0),
        rolling_days: (4, 6, 2),
        z_window: (40, 48, 8),
        pct_vol_lookback: (100, 100, 0),
        pct_min_sigma: (0.1, 0.1, 0.0),
        abs_vol_lookback: (100, 100, 0),
        session_mode: VwapDeviationSessionMode::RollingDays,
        use_close: true,
        deviation_mode: VwapDeviationMode::ZScore,
    })
}
