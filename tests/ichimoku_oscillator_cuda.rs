use vector_ta::indicators::ichimoku_oscillator::{
    ichimoku_oscillator_batch_with_kernel, IchimokuOscillatorBatchRange,
    IchimokuOscillatorNormalizeMode,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaIchimokuOscillator};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = vec![0.0; len];
    let mut low = vec![0.0; len];
    let mut close = vec![0.0; len];
    let mut source = vec![0.0; len];
    let mut base = 101.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.006).sin() * 0.25 + (x * 0.0014).cos() * 0.08;
        let c = base + (x * 0.018).sin() * 1.05 + (x * 0.012).cos() * 0.32;
        let span = 0.93 + (x * 0.014).sin().abs() * 0.35;
        close[i] = c;
        high[i] = c + span;
        low[i] = c - span * (0.8 + (x * 0.01).cos().abs() * 0.2);
        source[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    (high, low, close, source)
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
fn ichimoku_oscillator_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ichimoku_oscillator_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (high, low, close, source) = sample_ohlc(720);
    let sweep = IchimokuOscillatorBatchRange {
        conversion_periods: (8, 10, 2),
        base_periods: (24, 26, 2),
        lagging_span_periods: (48, 52, 4),
        displacement: (24, 24, 0),
        ma_length: (5, 5, 0),
        smoothing_length: (3, 5, 2),
        window_size: (40, 40, 0),
        top_band: (1.0, 1.0, 0.0),
        mid_band: (0.5, 0.5, 0.0),
        extra_smoothing: true,
        normalize: IchimokuOscillatorNormalizeMode::Window,
        clamp: true,
    };
    let cpu = ichimoku_oscillator_batch_with_kernel(
        &high,
        &low,
        &close,
        &source,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaIchimokuOscillator::new(0)?;
    let result = cuda.batch_dev(&high, &low, &close, &source, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());
    assert_device_matches(&cpu.signal, &result.outputs.signal.buf)?;
    assert_device_matches(&cpu.ma, &result.outputs.ma.buf)?;
    assert_device_matches(&cpu.conversion, &result.outputs.conversion.buf)?;
    assert_device_matches(&cpu.base, &result.outputs.base.buf)?;
    assert_device_matches(&cpu.chikou, &result.outputs.chikou.buf)?;
    assert_device_matches(&cpu.current_kumo_a, &result.outputs.current_kumo_a.buf)?;
    assert_device_matches(&cpu.current_kumo_b, &result.outputs.current_kumo_b.buf)?;
    assert_device_matches(&cpu.future_kumo_a, &result.outputs.future_kumo_a.buf)?;
    assert_device_matches(&cpu.future_kumo_b, &result.outputs.future_kumo_b.buf)?;
    assert_device_matches(&cpu.max_level, &result.outputs.max_level.buf)?;
    assert_device_matches(&cpu.high_level, &result.outputs.high_level.buf)?;
    assert_device_matches(&cpu.low_level, &result.outputs.low_level.buf)?;
    assert_device_matches(&cpu.min_level, &result.outputs.min_level.buf)?;
    Ok(())
}
