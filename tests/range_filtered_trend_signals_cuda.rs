use vector_ta::indicators::range_filtered_trend_signals::{
    range_filtered_trend_signals_batch_with_kernel, RangeFilteredTrendSignalsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRangeFilteredTrendSignals};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn range_filtered_trend_signals_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[range_filtered_trend_signals_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 960usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 103.0f64;
    for i in 24..len {
        let x = i as f64;
        base += (x * 0.008).sin() * 0.24 + (x * 0.0031).cos() * 0.11;
        close[i] = base + (x * 0.021).sin() * 0.82 + (x * 0.017).cos() * 0.29;
        let spread = 0.95 + (x * 0.014).cos().abs() * 0.37;
        high[i] = close[i] + spread;
        low[i] = close[i] - spread * (0.82 + (x * 0.009).sin().abs() * 0.22);
    }
    for i in 360..367 {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = RangeFilteredTrendSignalsBatchRange {
        kalman_alpha: (0.01, 0.02, 0.01),
        kalman_beta: (0.08, 0.10, 0.02),
        kalman_period: (77, 77, 0),
        dev: (1.1, 1.3, 0.2),
        supertrend_factor: (0.7, 0.7, 0.0),
        supertrend_atr_period: (7, 9, 2),
    };

    let cpu = range_filtered_trend_signals_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaRangeFilteredTrendSignals::new(0)?;
    let result = cuda.batch_dev(&high, &low, &close, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_kalman = vec![0.0f64; result.outputs.kalman.len()];
    let mut got_supertrend = vec![0.0f64; result.outputs.supertrend.len()];
    let mut got_upper_band = vec![0.0f64; result.outputs.upper_band.len()];
    let mut got_lower_band = vec![0.0f64; result.outputs.lower_band.len()];
    let mut got_trend = vec![0.0f64; result.outputs.trend.len()];
    let mut got_kalman_trend = vec![0.0f64; result.outputs.kalman_trend.len()];
    let mut got_state = vec![0.0f64; result.outputs.state.len()];
    let mut got_market_trending = vec![0.0f64; result.outputs.market_trending.len()];
    let mut got_market_ranging = vec![0.0f64; result.outputs.market_ranging.len()];
    let mut got_short_term_bullish = vec![0.0f64; result.outputs.short_term_bullish.len()];
    let mut got_short_term_bearish = vec![0.0f64; result.outputs.short_term_bearish.len()];
    let mut got_long_term_bullish = vec![0.0f64; result.outputs.long_term_bullish.len()];
    let mut got_long_term_bearish = vec![0.0f64; result.outputs.long_term_bearish.len()];
    result.outputs.kalman.buf.copy_to(&mut got_kalman)?;
    result.outputs.supertrend.buf.copy_to(&mut got_supertrend)?;
    result.outputs.upper_band.buf.copy_to(&mut got_upper_band)?;
    result.outputs.lower_band.buf.copy_to(&mut got_lower_band)?;
    result.outputs.trend.buf.copy_to(&mut got_trend)?;
    result
        .outputs
        .kalman_trend
        .buf
        .copy_to(&mut got_kalman_trend)?;
    result.outputs.state.buf.copy_to(&mut got_state)?;
    result
        .outputs
        .market_trending
        .buf
        .copy_to(&mut got_market_trending)?;
    result
        .outputs
        .market_ranging
        .buf
        .copy_to(&mut got_market_ranging)?;
    result
        .outputs
        .short_term_bullish
        .buf
        .copy_to(&mut got_short_term_bullish)?;
    result
        .outputs
        .short_term_bearish
        .buf
        .copy_to(&mut got_short_term_bearish)?;
    result
        .outputs
        .long_term_bullish
        .buf
        .copy_to(&mut got_long_term_bullish)?;
    result
        .outputs
        .long_term_bearish
        .buf
        .copy_to(&mut got_long_term_bearish)?;

    for idx in 0..cpu.kalman.len() {
        assert!(
            approx_eq(cpu.kalman[idx], got_kalman[idx], 1e-6),
            "kalman mismatch at {idx}: cpu={} cuda={}",
            cpu.kalman[idx],
            got_kalman[idx]
        );
        assert!(
            approx_eq(cpu.supertrend[idx], got_supertrend[idx], 1e-6),
            "supertrend mismatch at {idx}: cpu={} cuda={}",
            cpu.supertrend[idx],
            got_supertrend[idx]
        );
        assert!(
            approx_eq(cpu.upper_band[idx], got_upper_band[idx], 1e-6),
            "upper_band mismatch at {idx}: cpu={} cuda={}",
            cpu.upper_band[idx],
            got_upper_band[idx]
        );
        assert!(
            approx_eq(cpu.lower_band[idx], got_lower_band[idx], 1e-6),
            "lower_band mismatch at {idx}: cpu={} cuda={}",
            cpu.lower_band[idx],
            got_lower_band[idx]
        );
        assert!(
            approx_eq(cpu.trend[idx], got_trend[idx], 1e-6),
            "trend mismatch at {idx}: cpu={} cuda={}",
            cpu.trend[idx],
            got_trend[idx]
        );
        assert!(
            approx_eq(cpu.kalman_trend[idx], got_kalman_trend[idx], 1e-6),
            "kalman_trend mismatch at {idx}: cpu={} cuda={}",
            cpu.kalman_trend[idx],
            got_kalman_trend[idx]
        );
        assert!(
            approx_eq(cpu.state[idx], got_state[idx], 1e-6),
            "state mismatch at {idx}: cpu={} cuda={}",
            cpu.state[idx],
            got_state[idx]
        );
        assert!(
            approx_eq(cpu.market_trending[idx], got_market_trending[idx], 1e-6),
            "market_trending mismatch at {idx}: cpu={} cuda={}",
            cpu.market_trending[idx],
            got_market_trending[idx]
        );
        assert!(
            approx_eq(cpu.market_ranging[idx], got_market_ranging[idx], 1e-6),
            "market_ranging mismatch at {idx}: cpu={} cuda={}",
            cpu.market_ranging[idx],
            got_market_ranging[idx]
        );
        assert!(
            approx_eq(
                cpu.short_term_bullish[idx],
                got_short_term_bullish[idx],
                1e-6
            ),
            "short_term_bullish mismatch at {idx}: cpu={} cuda={}",
            cpu.short_term_bullish[idx],
            got_short_term_bullish[idx]
        );
        assert!(
            approx_eq(
                cpu.short_term_bearish[idx],
                got_short_term_bearish[idx],
                1e-6
            ),
            "short_term_bearish mismatch at {idx}: cpu={} cuda={}",
            cpu.short_term_bearish[idx],
            got_short_term_bearish[idx]
        );
        assert!(
            approx_eq(cpu.long_term_bullish[idx], got_long_term_bullish[idx], 1e-6),
            "long_term_bullish mismatch at {idx}: cpu={} cuda={}",
            cpu.long_term_bullish[idx],
            got_long_term_bullish[idx]
        );
        assert!(
            approx_eq(cpu.long_term_bearish[idx], got_long_term_bearish[idx], 1e-6),
            "long_term_bearish mismatch at {idx}: cpu={} cuda={}",
            cpu.long_term_bearish[idx],
            got_long_term_bearish[idx]
        );
    }

    Ok(())
}
