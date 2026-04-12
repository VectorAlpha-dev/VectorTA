use vector_ta::indicators::kase_peak_oscillator_with_divergences::{
    kase_peak_oscillator_with_divergences_batch_with_kernel,
    KasePeakOscillatorWithDivergencesBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaKasePeakOscillatorWithDivergences};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = vec![0.0; len];
    let mut low = vec![0.0; len];
    let mut close = vec![0.0; len];
    let mut base = 104.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.007).sin() * 0.21 + (x * 0.0013).cos() * 0.07;
        let c = base + (x * 0.021).sin() * 0.98 + (x * 0.013).cos() * 0.34;
        let span = 0.95 + (x * 0.016).sin().abs() * 0.38;
        close[i] = c;
        high[i] = c + span;
        low[i] = c - span * (0.78 + (x * 0.01).cos().abs() * 0.22);
    }
    (high, low, close)
}

#[cfg(feature = "cuda")]
fn assert_device_matches(
    expected: &[f64],
    buf: &DeviceBuffer<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut got = vec![0.0; expected.len()];
    buf.copy_to(&mut got)?;
    for idx in 0..expected.len() {
        assert!(approx_eq(expected[idx], got[idx], 1e-9));
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
fn kase_peak_oscillator_with_divergences_cuda_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[kase_peak_oscillator_with_divergences_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (high, low, close) = sample_ohlc(720);
    let sweep = KasePeakOscillatorWithDivergencesBatchRange {
        deviations: (2.0, 2.5, 0.5),
        short_cycle: (6, 8, 2),
        long_cycle: (18, 20, 2),
        sensitivity: (1.0, 1.5, 0.5),
        ..KasePeakOscillatorWithDivergencesBatchRange::default()
    };
    let cpu = kase_peak_oscillator_with_divergences_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaKasePeakOscillatorWithDivergences::new(0)?;
    let result = cuda.batch_dev(&high, &low, &close, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.deviations, cpu.deviations);
    assert_eq!(result.short_cycles, cpu.short_cycles);
    assert_eq!(result.long_cycles, cpu.long_cycles);
    assert_eq!(result.sensitivities, cpu.sensitivities);
    assert_device_matches(&cpu.oscillator, &result.outputs.oscillator.buf)?;
    assert_device_matches(&cpu.histogram, &result.outputs.histogram.buf)?;
    assert_device_matches(&cpu.max_peak_value, &result.outputs.max_peak_value.buf)?;
    assert_device_matches(&cpu.min_peak_value, &result.outputs.min_peak_value.buf)?;
    assert_device_matches(&cpu.market_extreme, &result.outputs.market_extreme.buf)?;
    assert_device_matches(&cpu.regular_bullish, &result.outputs.regular_bullish.buf)?;
    assert_device_matches(&cpu.hidden_bullish, &result.outputs.hidden_bullish.buf)?;
    assert_device_matches(&cpu.regular_bearish, &result.outputs.regular_bearish.buf)?;
    assert_device_matches(&cpu.hidden_bearish, &result.outputs.hidden_bearish.buf)?;
    assert_device_matches(&cpu.go_long, &result.outputs.go_long.buf)?;
    assert_device_matches(&cpu.go_short, &result.outputs.go_short.buf)?;
    Ok(())
}
