use vector_ta::indicators::smoothed_gaussian_trend_filter::{
    smoothed_gaussian_trend_filter_batch_with_kernel, SmoothedGaussianTrendFilterBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaSmoothedGaussianTrendFilter};

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
fn smoothed_gaussian_trend_filter_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[smoothed_gaussian_trend_filter_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 1984usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 104.0f64;
    for i in 8..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.42 + (x * 0.003).cos() * 0.18;
        let center = base + (x * 0.015).sin() * 0.53;
        high[i] = center + 0.78 + (x * 0.011).cos().abs() * 0.21;
        low[i] = center - 0.81 - (x * 0.013).sin().abs() * 0.19;
        close[i] = center + (x * 0.016).sin() * 0.24 - (x * 0.006).cos() * 0.07;
    }
    for i in (560..620).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1420..1490).step_by(9) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = SmoothedGaussianTrendFilterBatchRange {
        gaussian_length: (14, 16, 2),
        poles: (2, 4, 2),
        smoothing_length: (20, 24, 2),
        linreg_offset: (5, 7, 2),
    };
    let cpu = smoothed_gaussian_trend_filter_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda =
        CudaSmoothedGaussianTrendFilter::new(0).expect("CudaSmoothedGaussianTrendFilter::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_filter = vec![0.0f64; result.outputs.filter.len()];
    let mut got_supertrend = vec![0.0f64; result.outputs.supertrend.len()];
    let mut got_trend = vec![0.0f64; result.outputs.trend.len()];
    let mut got_ranging = vec![0.0f64; result.outputs.ranging.len()];
    result.outputs.filter.buf.copy_to(&mut got_filter)?;
    result.outputs.supertrend.buf.copy_to(&mut got_supertrend)?;
    result.outputs.trend.buf.copy_to(&mut got_trend)?;
    result.outputs.ranging.buf.copy_to(&mut got_ranging)?;

    for idx in 0..cpu.filter.len() {
        assert!(
            approx_eq(cpu.filter[idx], got_filter[idx], 1e-6),
            "filter mismatch at {idx}: cpu={} cuda={}",
            cpu.filter[idx],
            got_filter[idx]
        );
        assert!(
            approx_eq(cpu.supertrend[idx], got_supertrend[idx], 1e-6),
            "supertrend mismatch at {idx}: cpu={} cuda={}",
            cpu.supertrend[idx],
            got_supertrend[idx]
        );
        assert!(
            approx_eq(cpu.trend[idx], got_trend[idx], 1e-12),
            "trend mismatch at {idx}: cpu={} cuda={}",
            cpu.trend[idx],
            got_trend[idx]
        );
        assert!(
            approx_eq(cpu.ranging[idx], got_ranging[idx], 1e-12),
            "ranging mismatch at {idx}: cpu={} cuda={}",
            cpu.ranging[idx],
            got_ranging[idx]
        );
    }

    Ok(())
}
