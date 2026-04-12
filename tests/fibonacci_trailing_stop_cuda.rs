use vector_ta::indicators::fibonacci_trailing_stop::{
    fibonacci_trailing_stop_batch_with_kernel, FibonacciTrailingStopBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaFibonacciTrailingStop};

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
fn fibonacci_trailing_stop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[fibonacci_trailing_stop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2200usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 104.0f64;
    for i in 24..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.29 + (x * 0.004).cos() * 0.17;
        close[i] = base + (x * 0.019).sin() * 0.73 + (x * 0.006).cos() * 0.28;
        high[i] = close[i] + 1.06 + (x * 0.012).sin().abs() * 0.31;
        low[i] = close[i] - 0.98 - (x * 0.015).cos().abs() * 0.27;
    }
    for i in (520..620).step_by(13) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1460..1540).step_by(9) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = FibonacciTrailingStopBatchRange {
        left_bars: (5, 7, 2),
        right_bars: (1, 2, 1),
        level: (-0.618, -0.382, 0.236),
        trigger: Some("wick".to_string()),
    };
    let cpu = fibonacci_trailing_stop_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaFibonacciTrailingStop::new(0).expect("CudaFibonacciTrailingStop::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_trailing_stop = vec![0.0f64; result.outputs.trailing_stop.len()];
    let mut got_long_stop = vec![0.0f64; result.outputs.long_stop.len()];
    let mut got_short_stop = vec![0.0f64; result.outputs.short_stop.len()];
    let mut got_direction = vec![0.0f64; result.outputs.direction.len()];
    result
        .outputs
        .trailing_stop
        .buf
        .copy_to(&mut got_trailing_stop)?;
    result.outputs.long_stop.buf.copy_to(&mut got_long_stop)?;
    result.outputs.short_stop.buf.copy_to(&mut got_short_stop)?;
    result.outputs.direction.buf.copy_to(&mut got_direction)?;

    for idx in 0..cpu.trailing_stop.len() {
        assert!(
            approx_eq(cpu.trailing_stop[idx], got_trailing_stop[idx], 1e-9),
            "trailing_stop mismatch at {idx}: cpu={} cuda={}",
            cpu.trailing_stop[idx],
            got_trailing_stop[idx]
        );
        assert!(
            approx_eq(cpu.long_stop[idx], got_long_stop[idx], 1e-9),
            "long_stop mismatch at {idx}: cpu={} cuda={}",
            cpu.long_stop[idx],
            got_long_stop[idx]
        );
        assert!(
            approx_eq(cpu.short_stop[idx], got_short_stop[idx], 1e-9),
            "short_stop mismatch at {idx}: cpu={} cuda={}",
            cpu.short_stop[idx],
            got_short_stop[idx]
        );
        assert!(
            approx_eq(cpu.direction[idx], got_direction[idx], 1e-9),
            "direction mismatch at {idx}: cpu={} cuda={}",
            cpu.direction[idx],
            got_direction[idx]
        );
    }

    Ok(())
}
