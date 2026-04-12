use vector_ta::indicators::trend_follower::{
    trend_follower_batch_with_kernel, TrendFollowerBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaTrendFollower};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_hlcv(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut prev = 96.0f64;

    for i in 24..len {
        let x = i as f64;
        let c = prev + (x * 0.031).sin() * 0.91 + (x * 0.007).cos() * 0.24;
        let spread = 0.62 + (x * 0.015).sin().abs() * 0.26;
        high[i] = c + spread + 0.14;
        low[i] = c - spread - 0.13;
        close[i] = c;
        volume[i] = 1200.0 + (i % 90) as f64 * 11.0 + (x * 0.021).cos().abs() * 250.0;
        prev = c;
    }

    for i in (310..360).step_by(9) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }
    for i in (910..970).step_by(11) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    (high, low, close, volume)
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn trend_follower_cuda_batch_matches_cpu_with_linreg() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trend_follower_cuda_batch_matches_cpu_with_linreg] skipped - no CUDA device");
        return Ok(());
    }

    let (high, low, close, volume) = sample_hlcv(1536);
    let sweep = TrendFollowerBatchRange {
        trend_period: (18, 18, 0),
        ma_period: (12, 16, 4),
        channel_rate_percent: (1.0, 1.5, 0.5),
        linear_regression_period: (4, 6, 2),
        matype: ("ema".to_string(), "vwma".to_string(), String::new()),
        use_linear_regression: true,
    };
    let cpu = trend_follower_batch_with_kernel(
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaTrendFollower::new(0).expect("CudaTrendFollower::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &volume, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_values = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got_values)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got_values[idx], 1e-9),
            "values mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got_values[idx]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn trend_follower_cuda_batch_matches_cpu_without_linreg() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[trend_follower_cuda_batch_matches_cpu_without_linreg] skipped - no CUDA device"
        );
        return Ok(());
    }

    let (high, low, close, volume) = sample_hlcv(1536);
    let sweep = TrendFollowerBatchRange {
        trend_period: (14, 18, 4),
        ma_period: (10, 10, 0),
        channel_rate_percent: (0.8, 0.8, 0.0),
        linear_regression_period: (5, 5, 0),
        matype: ("sma".to_string(), "rma".to_string(), String::new()),
        use_linear_regression: false,
    };
    let cpu = trend_follower_batch_with_kernel(
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaTrendFollower::new(0).expect("CudaTrendFollower::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &volume, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_values = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got_values)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got_values[idx], 1e-9),
            "values mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got_values[idx]
        );
    }

    Ok(())
}
