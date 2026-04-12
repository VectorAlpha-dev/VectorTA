use vector_ta::indicators::ict_propulsion_block::{
    ict_propulsion_block_batch_with_kernel, IctPropulsionBlockBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaIctPropulsionBlock};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut open = vec![0.0; len];
    let mut high = vec![0.0; len];
    let mut low = vec![0.0; len];
    let mut close = vec![0.0; len];
    let mut base = 100.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.007).sin() * 0.26 + (x * 0.0013).cos() * 0.09;
        let c = base + (x * 0.02).sin() * 0.94 + (x * 0.011).cos() * 0.29;
        let o = c - (x * 0.017).cos() * 0.36;
        let span = 0.89 + (x * 0.015).sin().abs() * 0.33;
        open[i] = o;
        close[i] = c;
        high[i] = o.max(c) + span;
        low[i] = o.min(c) - span * (0.81 + (x * 0.01).cos().abs() * 0.19);
    }
    (open, high, low, close)
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
fn ict_propulsion_block_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ict_propulsion_block_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (open, high, low, close) = sample_ohlc(640);
    let sweep = IctPropulsionBlockBatchRange {
        swing_length: (5, 7, 2),
        mitigation_price: (true, false),
    };
    let cpu = ict_propulsion_block_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaIctPropulsionBlock::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());
    assert_device_matches(&cpu.bullish_high, &result.outputs.bullish_high.buf)?;
    assert_device_matches(&cpu.bullish_low, &result.outputs.bullish_low.buf)?;
    assert_device_matches(&cpu.bullish_kind, &result.outputs.bullish_kind.buf)?;
    assert_device_matches(&cpu.bullish_active, &result.outputs.bullish_active.buf)?;
    assert_device_matches(&cpu.bullish_mitigated, &result.outputs.bullish_mitigated.buf)?;
    assert_device_matches(&cpu.bullish_new, &result.outputs.bullish_new.buf)?;
    assert_device_matches(&cpu.bearish_high, &result.outputs.bearish_high.buf)?;
    assert_device_matches(&cpu.bearish_low, &result.outputs.bearish_low.buf)?;
    assert_device_matches(&cpu.bearish_kind, &result.outputs.bearish_kind.buf)?;
    assert_device_matches(&cpu.bearish_active, &result.outputs.bearish_active.buf)?;
    assert_device_matches(&cpu.bearish_mitigated, &result.outputs.bearish_mitigated.buf)?;
    assert_device_matches(&cpu.bearish_new, &result.outputs.bearish_new.buf)?;
    Ok(())
}
