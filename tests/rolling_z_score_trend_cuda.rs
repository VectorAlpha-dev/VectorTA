use vector_ta::indicators::rolling_z_score_trend::{
    rolling_z_score_trend_batch_with_kernel, RollingZScoreTrendBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRollingZScoreTrend};

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
fn rolling_z_score_trend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rolling_z_score_trend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 12..len {
        let x = i as f64;
        data[i] = 50.0 + 0.03 * x + (x * 0.014).sin() * 1.8 + (x * 0.009).cos() * 0.6;
    }
    for i in (1100..1180).step_by(9) {
        data[i] = f64::NAN;
    }

    let sweep = RollingZScoreTrendBatchRange {
        lookback_period: (12, 24, 6),
    };
    let cpu = rolling_z_score_trend_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaRollingZScoreTrend::new(0).expect("CudaRollingZScoreTrend::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_zscore = vec![0f64; result.outputs.zscore.len()];
    let mut got_momentum = vec![0f64; result.outputs.momentum.len()];
    result.outputs.zscore.buf.copy_to(&mut got_zscore)?;
    result.outputs.momentum.buf.copy_to(&mut got_momentum)?;

    for idx in 0..cpu.zscore.len() {
        assert!(
            approx_eq(cpu.zscore[idx], got_zscore[idx], 1e-3),
            "zscore mismatch at {idx}: cpu={} cuda={}",
            cpu.zscore[idx],
            got_zscore[idx]
        );
        assert!(
            approx_eq(cpu.momentum[idx], got_momentum[idx], 1e-3),
            "momentum mismatch at {idx}: cpu={} cuda={}",
            cpu.momentum[idx],
            got_momentum[idx]
        );
    }

    Ok(())
}
