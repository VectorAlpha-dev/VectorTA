use vector_ta::indicators::trend_continuation_factor::{
    trend_continuation_factor_batch_with_kernel, TrendContinuationFactorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaTrendContinuationFactor};

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
fn trend_continuation_factor_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trend_continuation_factor_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 3584usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 7..len {
        value += (i as f64 * 0.013).sin() * 0.72 + (i as f64 * 0.005).cos() * 0.31;
        data[i] = value + (i as f64 * 0.003).sin() * 0.14;
    }
    for i in (900..980).step_by(11) {
        data[i] = f64::NAN;
    }
    for i in (2200..2280).step_by(9) {
        data[i] = f64::NAN;
    }

    let sweep = TrendContinuationFactorBatchRange {
        length: (12, 20, 4),
    };
    let cpu = trend_continuation_factor_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaTrendContinuationFactor::new(0).expect("CudaTrendContinuationFactor::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_plus = vec![0f64; result.outputs.plus_tcf.len()];
    let mut got_minus = vec![0f64; result.outputs.minus_tcf.len()];
    result.outputs.plus_tcf.buf.copy_to(&mut got_plus)?;
    result.outputs.minus_tcf.buf.copy_to(&mut got_minus)?;

    for idx in 0..cpu.plus_tcf.len() {
        assert!(
            approx_eq(cpu.plus_tcf[idx], got_plus[idx], 1e-10),
            "plus mismatch at {idx}: cpu={} cuda={}",
            cpu.plus_tcf[idx],
            got_plus[idx]
        );
        assert!(
            approx_eq(cpu.minus_tcf[idx], got_minus[idx], 1e-10),
            "minus mismatch at {idx}: cpu={} cuda={}",
            cpu.minus_tcf[idx],
            got_minus[idx]
        );
    }

    Ok(())
}
