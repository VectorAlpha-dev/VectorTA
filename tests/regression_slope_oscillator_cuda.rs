use vector_ta::indicators::regression_slope_oscillator::{
    regression_slope_oscillator_batch_with_kernel, RegressionSlopeOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRegressionSlopeOscillator};

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
fn regression_slope_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[regression_slope_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1856usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 83.0f64;
    for i in 10..len {
        let x = i as f64;
        base *= 1.0 + (x * 0.0011).sin() * 0.0018 + (x * 0.0007).cos() * 0.0009;
        data[i] = base + 0.35 + (x * 0.017).sin().abs() * 0.41 + (x * 0.004).cos().abs() * 0.12;
    }
    for i in (420..490).step_by(11) {
        data[i] = f64::NAN;
    }
    for i in (1180..1260).step_by(13) {
        data[i] = f64::NAN;
    }

    let sweep = RegressionSlopeOscillatorBatchRange {
        min_range: (10, 14, 2),
        max_range: (28, 32, 2),
        step: (4, 4, 0),
        signal_line: (5, 7, 2),
    };
    let cpu = regression_slope_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaRegressionSlopeOscillator::new(0).expect("CudaRegressionSlopeOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_value = vec![0.0f64; result.outputs.value.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    let mut got_bullish = vec![0.0f64; result.outputs.bullish_reversal.len()];
    let mut got_bearish = vec![0.0f64; result.outputs.bearish_reversal.len()];
    result.outputs.value.buf.copy_to(&mut got_value)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;
    result
        .outputs
        .bullish_reversal
        .buf
        .copy_to(&mut got_bullish)?;
    result
        .outputs
        .bearish_reversal
        .buf
        .copy_to(&mut got_bearish)?;

    for idx in 0..cpu.value.len() {
        assert!(
            approx_eq(cpu.value[idx], got_value[idx], 1e-9),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.value[idx],
            got_value[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-9),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
        assert!(
            approx_eq(cpu.bullish_reversal[idx], got_bullish[idx], 1e-12),
            "bullish mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_reversal[idx],
            got_bullish[idx]
        );
        assert!(
            approx_eq(cpu.bearish_reversal[idx], got_bearish[idx], 1e-12),
            "bearish mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_reversal[idx],
            got_bearish[idx]
        );
    }

    Ok(())
}
