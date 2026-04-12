use vector_ta::indicators::neighboring_trailing_stop::{
    neighboring_trailing_stop_batch_with_kernel, NeighboringTrailingStopBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaNeighboringTrailingStop};

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
fn neighboring_trailing_stop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[neighboring_trailing_stop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2400usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 121.0f64;
    for i in 25..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.27 + (x * 0.003).cos() * 0.15;
        close[i] = base + (x * 0.018).sin() * 0.88 + (x * 0.007).cos() * 0.24;
        high[i] = close[i] + 0.93 + (x * 0.012).sin().abs() * 0.29;
        low[i] = close[i] - 0.86 - (x * 0.015).cos().abs() * 0.26;
    }
    for i in (860..930).step_by(14) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1710..1780).step_by(11) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = NeighboringTrailingStopBatchRange {
        buffer_size: (120, 160, 40),
        k: (8, 12, 4),
        percentile: (80.0, 90.0, 10.0),
        smooth: (3, 5, 2),
    };
    let cpu = neighboring_trailing_stop_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaNeighboringTrailingStop::new(0).expect("CudaNeighboringTrailingStop::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_trailing_stop = vec![0.0f64; result.outputs.trailing_stop.len()];
    let mut got_bullish_band = vec![0.0f64; result.outputs.bullish_band.len()];
    let mut got_bearish_band = vec![0.0f64; result.outputs.bearish_band.len()];
    let mut got_direction = vec![0.0f64; result.outputs.direction.len()];
    let mut got_discovery_bull = vec![0.0f64; result.outputs.discovery_bull.len()];
    let mut got_discovery_bear = vec![0.0f64; result.outputs.discovery_bear.len()];
    result
        .outputs
        .trailing_stop
        .buf
        .copy_to(&mut got_trailing_stop)?;
    result
        .outputs
        .bullish_band
        .buf
        .copy_to(&mut got_bullish_band)?;
    result
        .outputs
        .bearish_band
        .buf
        .copy_to(&mut got_bearish_band)?;
    result.outputs.direction.buf.copy_to(&mut got_direction)?;
    result
        .outputs
        .discovery_bull
        .buf
        .copy_to(&mut got_discovery_bull)?;
    result
        .outputs
        .discovery_bear
        .buf
        .copy_to(&mut got_discovery_bear)?;

    for idx in 0..cpu.trailing_stop.len() {
        assert!(
            approx_eq(cpu.trailing_stop[idx], got_trailing_stop[idx], 1e-9),
            "trailing_stop mismatch at {idx}: cpu={} cuda={}",
            cpu.trailing_stop[idx],
            got_trailing_stop[idx]
        );
        assert!(
            approx_eq(cpu.bullish_band[idx], got_bullish_band[idx], 1e-9),
            "bullish_band mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_band[idx],
            got_bullish_band[idx]
        );
        assert!(
            approx_eq(cpu.bearish_band[idx], got_bearish_band[idx], 1e-9),
            "bearish_band mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_band[idx],
            got_bearish_band[idx]
        );
        assert!(
            approx_eq(cpu.direction[idx], got_direction[idx], 1e-9),
            "direction mismatch at {idx}: cpu={} cuda={}",
            cpu.direction[idx],
            got_direction[idx]
        );
        assert!(
            approx_eq(cpu.discovery_bull[idx], got_discovery_bull[idx], 1e-9),
            "discovery_bull mismatch at {idx}: cpu={} cuda={}",
            cpu.discovery_bull[idx],
            got_discovery_bull[idx]
        );
        assert!(
            approx_eq(cpu.discovery_bear[idx], got_discovery_bear[idx], 1e-9),
            "discovery_bear mismatch at {idx}: cpu={} cuda={}",
            cpu.discovery_bear[idx],
            got_discovery_bear[idx]
        );
    }

    Ok(())
}
