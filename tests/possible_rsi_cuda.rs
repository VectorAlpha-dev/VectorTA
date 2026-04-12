use vector_ta::indicators::possible_rsi::{
    possible_rsi_batch_with_kernel, PossibleRsiBatchRange, PossibleRsiParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaPossibleRsi};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_data(len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    let mut base = 102.0f64;
    for (i, value) in out.iter_mut().enumerate() {
        let x = i as f64;
        base += (x * 0.007).sin() * 0.22 + (x * 0.0014).cos() * 0.06;
        *value = base + (x * 0.025).sin() * 0.88 + (x * 0.013).cos() * 0.27;
    }
    out
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
fn possible_rsi_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[possible_rsi_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let data = sample_data(720);
    let range = PossibleRsiBatchRange {
        period: (24, 28, 4),
        highpass_period: (10, 12, 2),
        ..PossibleRsiBatchRange::default()
    };
    let base = PossibleRsiParams {
        rsi_mode: Some("rsx".to_string()),
        normalization_mode: Some("softmax".to_string()),
        signal_type: Some("levels_crossover".to_string()),
        run_highpass: Some(true),
        ..PossibleRsiParams::default()
    };
    let cpu = possible_rsi_batch_with_kernel(&data, &range, &base, Kernel::ScalarBatch)?;
    let cuda = CudaPossibleRsi::new(0)?;
    let result = cuda.batch_dev(&data, &range, &base)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());
    assert_device_matches(&cpu.value, &result.outputs.value.buf)?;
    assert_device_matches(&cpu.buy_level, &result.outputs.buy_level.buf)?;
    assert_device_matches(&cpu.sell_level, &result.outputs.sell_level.buf)?;
    assert_device_matches(&cpu.middle_level, &result.outputs.middle_level.buf)?;
    assert_device_matches(&cpu.state, &result.outputs.state.buf)?;
    assert_device_matches(&cpu.long_signal, &result.outputs.long_signal.buf)?;
    assert_device_matches(&cpu.short_signal, &result.outputs.short_signal.buf)?;
    Ok(())
}
