use vector_ta::indicators::nonlinear_regression_zero_lag_moving_average::{
    nonlinear_regression_zero_lag_moving_average_batch_with_kernel,
    NonlinearRegressionZeroLagMovingAverageBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaNonlinearRegressionZeroLagMovingAverage};

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
fn nonlinear_regression_zero_lag_moving_average_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[nonlinear_regression_zero_lag_moving_average_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2240usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 108.0f64;
    for i in 20..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.42 + (x * 0.003).cos() * 0.18;
        data[i] = base + (x * 0.017).sin() * 0.59 + (x * 0.006).cos() * 0.21;
    }
    for i in (520..600).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1480..1560).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = NonlinearRegressionZeroLagMovingAverageBatchRange {
        zlma_period: (11, 15, 4),
        regression_period: (9, 13, 4),
    };
    let cpu = nonlinear_regression_zero_lag_moving_average_batch_with_kernel(
        &data,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaNonlinearRegressionZeroLagMovingAverage::new(0)
        .expect("CudaNonlinearRegressionZeroLagMovingAverage::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_value = vec![0.0f64; result.outputs.value.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    let mut got_long = vec![0.0f64; result.outputs.long_signal.len()];
    let mut got_short = vec![0.0f64; result.outputs.short_signal.len()];
    result.outputs.value.buf.copy_to(&mut got_value)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;
    result.outputs.long_signal.buf.copy_to(&mut got_long)?;
    result.outputs.short_signal.buf.copy_to(&mut got_short)?;

    for idx in 0..cpu.value.len() {
        assert!(
            approx_eq(cpu.value[idx], got_value[idx], 1e-6),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.value[idx],
            got_value[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-6),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
        assert!(
            approx_eq(cpu.long_signal[idx], got_long[idx], 1e-6),
            "long_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.long_signal[idx],
            got_long[idx]
        );
        assert!(
            approx_eq(cpu.short_signal[idx], got_short[idx], 1e-6),
            "short_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.short_signal[idx],
            got_short[idx]
        );
    }

    Ok(())
}
