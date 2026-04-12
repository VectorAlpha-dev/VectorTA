use vector_ta::indicators::market_structure_confluence::{
    market_structure_confluence_batch_with_kernel, MarketStructureConfluenceBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMarketStructureConfluence};

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
    let mut base = 99.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.28 + (x * 0.0019).cos() * 0.08;
        let c = base + (x * 0.018).sin() * 1.01 + (x * 0.012).cos() * 0.36;
        let span = 0.91 + (x * 0.014).sin().abs() * 0.41;
        close[i] = c;
        high[i] = c + span;
        low[i] = c - span * (0.82 + (x * 0.009).cos().abs() * 0.18);
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
fn market_structure_confluence_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[market_structure_confluence_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (high, low, close) = sample_ohlc(640);
    let sweep = MarketStructureConfluenceBatchRange {
        swing_size: (8, 10, 2),
        bos_confirmation: vec!["Candle Close".to_string(), "Wicks".to_string()],
        vol_mult: (1.5, 2.0, 0.5),
        ..MarketStructureConfluenceBatchRange::default()
    };
    let cpu =
        market_structure_confluence_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaMarketStructureConfluence::new(0)?;
    let result = cuda.batch_dev(&high, &low, &close, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());
    assert_device_matches(&cpu.basis, &result.outputs.basis.buf)?;
    assert_device_matches(&cpu.upper_band, &result.outputs.upper_band.buf)?;
    assert_device_matches(&cpu.lower_band, &result.outputs.lower_band.buf)?;
    assert_device_matches(&cpu.structure_direction, &result.outputs.structure_direction.buf)?;
    assert_device_matches(&cpu.bullish_arrow, &result.outputs.bullish_arrow.buf)?;
    assert_device_matches(&cpu.bearish_arrow, &result.outputs.bearish_arrow.buf)?;
    assert_device_matches(&cpu.bullish_change, &result.outputs.bullish_change.buf)?;
    assert_device_matches(&cpu.bearish_change, &result.outputs.bearish_change.buf)?;
    assert_device_matches(&cpu.hh, &result.outputs.hh.buf)?;
    assert_device_matches(&cpu.lh, &result.outputs.lh.buf)?;
    assert_device_matches(&cpu.hl, &result.outputs.hl.buf)?;
    assert_device_matches(&cpu.ll, &result.outputs.ll.buf)?;
    assert_device_matches(&cpu.bullish_bos, &result.outputs.bullish_bos.buf)?;
    assert_device_matches(&cpu.bullish_choch, &result.outputs.bullish_choch.buf)?;
    assert_device_matches(&cpu.bearish_bos, &result.outputs.bearish_bos.buf)?;
    assert_device_matches(&cpu.bearish_choch, &result.outputs.bearish_choch.buf)?;
    Ok(())
}
