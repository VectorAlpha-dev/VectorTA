use vector_ta::indicators::exponential_trend::{
    exponential_trend_batch_with_kernel, ExponentialTrendBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaExponentialTrend};

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
fn exponential_trend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[exponential_trend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 448usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 110.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.38 + (x * 0.003).cos() * 0.14;
        let c = base + (x * 0.019).sin() * 1.1 + (x * 0.006).cos() * 0.5;
        let span = 1.7 + (x * 0.017).sin().abs() * 0.9;
        high[i] = c + span * 0.58;
        low[i] = c - span * 0.42;
        close[i] = c;
    }
    for i in 180..186 {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in 326..331 {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = ExponentialTrendBatchRange {
        exp_rate: (0.00003, 0.00005, 0.00002),
        initial_distance: (4.0, 5.0, 1.0),
        width_multiplier: (1.0, 1.5, 0.5),
    };
    let cpu =
        exponential_trend_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaExponentialTrend::new(0).expect("CudaExponentialTrend::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_uptrend_base = vec![0.0f64; result.outputs.uptrend_base.len()];
    let mut got_downtrend_base = vec![0.0f64; result.outputs.downtrend_base.len()];
    let mut got_uptrend_extension = vec![0.0f64; result.outputs.uptrend_extension.len()];
    let mut got_downtrend_extension = vec![0.0f64; result.outputs.downtrend_extension.len()];
    let mut got_bullish_change = vec![0.0f64; result.outputs.bullish_change.len()];
    let mut got_bearish_change = vec![0.0f64; result.outputs.bearish_change.len()];
    result
        .outputs
        .uptrend_base
        .buf
        .copy_to(&mut got_uptrend_base)?;
    result
        .outputs
        .downtrend_base
        .buf
        .copy_to(&mut got_downtrend_base)?;
    result
        .outputs
        .uptrend_extension
        .buf
        .copy_to(&mut got_uptrend_extension)?;
    result
        .outputs
        .downtrend_extension
        .buf
        .copy_to(&mut got_downtrend_extension)?;
    result
        .outputs
        .bullish_change
        .buf
        .copy_to(&mut got_bullish_change)?;
    result
        .outputs
        .bearish_change
        .buf
        .copy_to(&mut got_bearish_change)?;

    for idx in 0..cpu.uptrend_base.len() {
        assert!(
            approx_eq(cpu.uptrend_base[idx], got_uptrend_base[idx], 1e-6),
            "uptrend_base mismatch at {idx}: cpu={} cuda={}",
            cpu.uptrend_base[idx],
            got_uptrend_base[idx]
        );
        assert!(
            approx_eq(cpu.downtrend_base[idx], got_downtrend_base[idx], 1e-6),
            "downtrend_base mismatch at {idx}: cpu={} cuda={}",
            cpu.downtrend_base[idx],
            got_downtrend_base[idx]
        );
        assert!(
            approx_eq(cpu.uptrend_extension[idx], got_uptrend_extension[idx], 1e-6),
            "uptrend_extension mismatch at {idx}: cpu={} cuda={}",
            cpu.uptrend_extension[idx],
            got_uptrend_extension[idx]
        );
        assert!(
            approx_eq(
                cpu.downtrend_extension[idx],
                got_downtrend_extension[idx],
                1e-6
            ),
            "downtrend_extension mismatch at {idx}: cpu={} cuda={}",
            cpu.downtrend_extension[idx],
            got_downtrend_extension[idx]
        );
        assert!(
            approx_eq(cpu.bullish_change[idx], got_bullish_change[idx], 1e-6),
            "bullish_change mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_change[idx],
            got_bullish_change[idx]
        );
        assert!(
            approx_eq(cpu.bearish_change[idx], got_bearish_change[idx], 1e-6),
            "bearish_change mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_change[idx],
            got_bearish_change[idx]
        );
    }

    Ok(())
}
