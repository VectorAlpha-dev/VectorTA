use vector_ta::indicators::reversal_signals::{
    reversal_signals_batch_with_kernel, ReversalSignalsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaReversalSignals};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlcv(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 24..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.28 + (x * 0.003).cos() * 0.09;
        let c = base + (x * 0.047).sin() * 1.75 + (x * 0.019).cos() * 0.46;
        let o = c - (x * 0.031).cos() * 0.63;
        let spread = 0.95 + (x * 0.013).sin().abs() * 0.41;
        open[i] = o;
        close[i] = c;
        high[i] = o.max(c) + spread;
        low[i] = o.min(c) - spread * (0.81 + (x * 0.011).cos().abs() * 0.24);
        volume[i] = 1100.0 + x * 5.0 + (x * 0.053).sin() * 220.0 + (x * 0.017).cos() * 90.0;
    }
    (open, high, low, close, volume)
}

#[cfg(feature = "cuda")]
fn run_case(sweep: ReversalSignalsBatchRange) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[reversal_signals_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (mut open, mut high, mut low, mut close, mut volume) = sample_ohlcv(520);
    for i in 311..315 {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let cpu = reversal_signals_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaReversalSignals::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &volume, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_buy = vec![0.0f64; result.outputs.buy_signal.len()];
    let mut got_sell = vec![0.0f64; result.outputs.sell_signal.len()];
    let mut got_stepped = vec![0.0f64; result.outputs.stepped_ma.len()];
    let mut got_state = vec![0.0f64; result.outputs.state.len()];
    result.outputs.buy_signal.buf.copy_to(&mut got_buy)?;
    result.outputs.sell_signal.buf.copy_to(&mut got_sell)?;
    result.outputs.stepped_ma.buf.copy_to(&mut got_stepped)?;
    result.outputs.state.buf.copy_to(&mut got_state)?;

    for idx in 0..cpu.buy_signal.len() {
        assert!(approx_eq(cpu.buy_signal[idx], got_buy[idx], 1e-6));
        assert!(approx_eq(cpu.sell_signal[idx], got_sell[idx], 1e-6));
        assert!(approx_eq(cpu.stepped_ma[idx], got_stepped[idx], 1e-6));
        assert!(approx_eq(cpu.state[idx], got_state[idx], 1e-6));
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
fn reversal_signals_cuda_ema_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(ReversalSignalsBatchRange {
        lookback_period: (10, 12, 2),
        confirmation_period: (2, 3, 1),
        trend_ma_period: (34, 38, 4),
        ma_step_period: (21, 25, 4),
        use_volume_confirmation: true,
        trend_ma_type: "EMA".to_string(),
    })
}

#[cfg(feature = "cuda")]
#[test]
fn reversal_signals_cuda_vwma_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_case(ReversalSignalsBatchRange {
        lookback_period: (9, 11, 2),
        confirmation_period: (2, 2, 0),
        trend_ma_period: (16, 20, 4),
        ma_step_period: (13, 17, 4),
        use_volume_confirmation: false,
        trend_ma_type: "VWMA".to_string(),
    })
}
