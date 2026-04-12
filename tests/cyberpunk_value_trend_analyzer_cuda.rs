use vector_ta::indicators::cyberpunk_value_trend_analyzer::{
    cyberpunk_value_trend_analyzer_batch_with_kernel, CyberpunkValueTrendAnalyzerBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaCyberpunkValueTrendAnalyzer};

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
fn cyberpunk_value_trend_analyzer_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[cyberpunk_value_trend_analyzer_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 640usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 112.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.42 + (x * 0.0031).cos() * 0.18;
        close[i] = base + (x * 0.061).sin() * 3.0 + (x * 0.019).cos() * 1.4;
        open[i] = close[i] - (x * 0.037).sin() * 0.9;
        let span = 1.4 + (x * 0.021).sin().abs() * 0.7;
        high[i] = close[i].max(open[i]) + span;
        low[i] = close[i].min(open[i]) - span * 0.9;
    }
    open[311] = f64::NAN;
    high[311] = f64::NAN;
    low[311] = f64::NAN;
    close[311] = f64::NAN;

    let sweep = CyberpunkValueTrendAnalyzerBatchRange {
        entry_level: (25, 35, 5),
        exit_level: (68, 84, 8),
    };

    let cpu = cyberpunk_value_trend_analyzer_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaCyberpunkValueTrendAnalyzer::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_value_trend = vec![0.0f64; result.outputs.value_trend.len()];
    let mut got_value_trend_lag = vec![0.0f64; result.outputs.value_trend_lag.len()];
    let mut got_deviation_index = vec![0.0f64; result.outputs.deviation_index.len()];
    let mut got_overbought_signal = vec![0.0f64; result.outputs.overbought_signal.len()];
    let mut got_buy_signal = vec![0.0f64; result.outputs.buy_signal.len()];
    let mut got_sell_signal = vec![0.0f64; result.outputs.sell_signal.len()];
    result
        .outputs
        .value_trend
        .buf
        .copy_to(&mut got_value_trend)?;
    result
        .outputs
        .value_trend_lag
        .buf
        .copy_to(&mut got_value_trend_lag)?;
    result
        .outputs
        .deviation_index
        .buf
        .copy_to(&mut got_deviation_index)?;
    result
        .outputs
        .overbought_signal
        .buf
        .copy_to(&mut got_overbought_signal)?;
    result.outputs.buy_signal.buf.copy_to(&mut got_buy_signal)?;
    result
        .outputs
        .sell_signal
        .buf
        .copy_to(&mut got_sell_signal)?;

    for idx in 0..cpu.value_trend.len() {
        assert!(
            approx_eq(cpu.value_trend[idx], got_value_trend[idx], 1e-6),
            "value_trend mismatch at {idx}: cpu={} cuda={}",
            cpu.value_trend[idx],
            got_value_trend[idx]
        );
        assert!(
            approx_eq(cpu.value_trend_lag[idx], got_value_trend_lag[idx], 1e-6),
            "value_trend_lag mismatch at {idx}: cpu={} cuda={}",
            cpu.value_trend_lag[idx],
            got_value_trend_lag[idx]
        );
        assert!(
            approx_eq(cpu.deviation_index[idx], got_deviation_index[idx], 1e-6),
            "deviation_index mismatch at {idx}: cpu={} cuda={}",
            cpu.deviation_index[idx],
            got_deviation_index[idx]
        );
        assert!(
            approx_eq(cpu.overbought_signal[idx], got_overbought_signal[idx], 1e-6),
            "overbought_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.overbought_signal[idx],
            got_overbought_signal[idx]
        );
        assert!(
            approx_eq(cpu.buy_signal[idx], got_buy_signal[idx], 1e-6),
            "buy_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.buy_signal[idx],
            got_buy_signal[idx]
        );
        assert!(
            approx_eq(cpu.sell_signal[idx], got_sell_signal[idx], 1e-6),
            "sell_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.sell_signal[idx],
            got_sell_signal[idx]
        );
    }

    Ok(())
}
