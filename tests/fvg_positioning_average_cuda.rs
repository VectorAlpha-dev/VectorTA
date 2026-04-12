use vector_ta::indicators::fvg_positioning_average::{
    fvg_positioning_average_batch_with_kernel, FvgPositioningAverageBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaFvgPositioningAverage};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn build_gap_series(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 96.0f64;

    for i in 14..len {
        let x = i as f64;
        base += (x * 0.017).sin() * 0.36 + (x * 0.004).cos() * 0.11;
        let center = base + (x * 0.043).sin() * 0.9;
        let body = (x * 0.029).sin() * 0.55;
        open[i] = center - body;
        close[i] = center + body * 0.85;
        let span = 0.95 + (x * 0.013).sin().abs() * 0.4;
        high[i] = open[i].max(close[i]) + span;
        low[i] = open[i].min(close[i]) - span * 0.9;
    }

    for &i in &[70usize, 118, 182, 246, 320, 388] {
        let ref_high = high[i - 2];
        close[i - 1] = ref_high + 1.6;
        open[i - 1] = close[i - 1] - 0.4;
        high[i - 1] = close[i - 1] + 0.7;
        low[i - 1] = open[i - 1] - 0.6;

        low[i] = ref_high + 2.4;
        open[i] = low[i] + 0.5;
        close[i] = low[i] + 1.4;
        high[i] = close[i] + 0.8;
    }

    for &i in &[94usize, 156, 214, 278, 350, 430] {
        let ref_low = low[i - 2];
        close[i - 1] = ref_low - 1.6;
        open[i - 1] = close[i - 1] + 0.4;
        high[i - 1] = open[i - 1] + 0.6;
        low[i - 1] = close[i - 1] - 0.7;

        high[i] = ref_low - 2.3;
        open[i] = high[i] - 0.4;
        close[i] = high[i] - 1.2;
        low[i] = close[i] - 0.8;
    }

    open[257] = f64::NAN;
    high[257] = f64::NAN;
    low[257] = f64::NAN;
    close[257] = f64::NAN;

    (open, high, low, close)
}

#[cfg(feature = "cuda")]
fn run_mode(mode: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[fvg_positioning_average_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (open, high, low, close) = build_gap_series(480);
    let sweep = FvgPositioningAverageBatchRange {
        lookback: (5, 8, 3),
        atr_multiplier: (0.15, 0.35, 0.20),
    };

    let cpu = fvg_positioning_average_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        mode,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaFvgPositioningAverage::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &sweep, mode)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_bull_average = vec![0.0f64; result.outputs.bull_average.len()];
    let mut got_bear_average = vec![0.0f64; result.outputs.bear_average.len()];
    let mut got_bull_mid = vec![0.0f64; result.outputs.bull_mid.len()];
    let mut got_bear_mid = vec![0.0f64; result.outputs.bear_mid.len()];
    result
        .outputs
        .bull_average
        .buf
        .copy_to(&mut got_bull_average)?;
    result
        .outputs
        .bear_average
        .buf
        .copy_to(&mut got_bear_average)?;
    result.outputs.bull_mid.buf.copy_to(&mut got_bull_mid)?;
    result.outputs.bear_mid.buf.copy_to(&mut got_bear_mid)?;

    for idx in 0..cpu.bull_average.len() {
        assert!(
            approx_eq(cpu.bull_average[idx], got_bull_average[idx], 1e-6),
            "bull_average mismatch at {idx}: cpu={} cuda={}",
            cpu.bull_average[idx],
            got_bull_average[idx]
        );
        assert!(
            approx_eq(cpu.bear_average[idx], got_bear_average[idx], 1e-6),
            "bear_average mismatch at {idx}: cpu={} cuda={}",
            cpu.bear_average[idx],
            got_bear_average[idx]
        );
        assert!(
            approx_eq(cpu.bull_mid[idx], got_bull_mid[idx], 1e-6),
            "bull_mid mismatch at {idx}: cpu={} cuda={}",
            cpu.bull_mid[idx],
            got_bull_mid[idx]
        );
        assert!(
            approx_eq(cpu.bear_mid[idx], got_bear_mid[idx], 1e-6),
            "bear_mid mismatch at {idx}: cpu={} cuda={}",
            cpu.bear_mid[idx],
            got_bear_mid[idx]
        );
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
fn fvg_positioning_average_cuda_bar_count_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_mode("Bar Count")
}

#[cfg(feature = "cuda")]
#[test]
fn fvg_positioning_average_cuda_fvg_count_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    run_mode("FVG Count")
}
