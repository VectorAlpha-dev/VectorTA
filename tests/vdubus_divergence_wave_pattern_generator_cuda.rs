use vector_ta::indicators::vdubus_divergence_wave_pattern_generator::{
    vdubus_divergence_wave_pattern_generator_batch_with_kernel,
    VdubusDivergenceWavePatternGeneratorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVdubusDivergenceWavePatternGenerator};

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
    let mut base = 103.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.006).sin() * 0.24 + (x * 0.0016).cos() * 0.08;
        let c = base + (x * 0.022).sin() * 1.08 + (x * 0.012).cos() * 0.37;
        let span = 0.94 + (x * 0.017).sin().abs() * 0.36;
        close[i] = c;
        high[i] = c + span;
        low[i] = c - span * (0.79 + (x * 0.01).cos().abs() * 0.21);
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
fn vdubus_divergence_wave_pattern_generator_cuda_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vdubus_divergence_wave_pattern_generator_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (high, low, close) = sample_ohlc(720);
    let sweep = VdubusDivergenceWavePatternGeneratorBatchRange {
        fast_depth: (4, 6, 2),
        slow_depth: (10, 12, 2),
        fast_length: (5, 5, 0),
        slow_length: (8, 8, 0),
        signal_length: (4, 6, 2),
        lookback: (20, 24, 4),
        err_tol: (0.12, 0.16, 0.04),
        ..VdubusDivergenceWavePatternGeneratorBatchRange::default()
    };
    let cpu = vdubus_divergence_wave_pattern_generator_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaVdubusDivergenceWavePatternGenerator::new(0)?;
    let result = cuda.batch_dev(&high, &low, &close, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());
    assert_device_matches(&cpu.fast_standard, &result.outputs.fast_standard.buf)?;
    assert_device_matches(&cpu.fast_climax, &result.outputs.fast_climax.buf)?;
    assert_device_matches(&cpu.fast_rounded, &result.outputs.fast_rounded.buf)?;
    assert_device_matches(&cpu.fast_predator, &result.outputs.fast_predator.buf)?;
    assert_device_matches(&cpu.slow_standard, &result.outputs.slow_standard.buf)?;
    assert_device_matches(&cpu.slow_climax, &result.outputs.slow_climax.buf)?;
    assert_device_matches(&cpu.slow_rounded, &result.outputs.slow_rounded.buf)?;
    assert_device_matches(&cpu.slow_predator, &result.outputs.slow_predator.buf)?;
    assert_device_matches(&cpu.opposing_force, &result.outputs.opposing_force.buf)?;
    assert_device_matches(&cpu.macd, &result.outputs.macd.buf)?;
    assert_device_matches(&cpu.signal, &result.outputs.signal.buf)?;
    assert_device_matches(&cpu.hist, &result.outputs.hist.buf)?;
    Ok(())
}
