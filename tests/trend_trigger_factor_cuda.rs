use vector_ta::indicators::trend_trigger_factor::{
    trend_trigger_factor_batch_with_kernel, TrendTriggerFactorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaTrendTriggerFactor};

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
fn trend_trigger_factor_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trend_trigger_factor_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 3072usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut base = 100.0f64;
    for i in 6..len {
        base += (i as f64 * 0.014).sin() * 0.63 + (i as f64 * 0.008).cos() * 0.21;
        let center = base + (i as f64 * 0.005).sin() * 0.19;
        high[i] = center + 1.45 + (i as f64 * 0.011).cos().abs() * 0.18;
        low[i] = center - 1.38 - (i as f64 * 0.007).sin().abs() * 0.16;
    }
    for i in (1200..1265).step_by(9) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
    }
    for i in (2100..2148).step_by(12) {
        high[i] = f64::NAN;
    }

    let sweep = TrendTriggerFactorBatchRange {
        length: (15, 19, 2),
    };
    let cpu = trend_trigger_factor_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaTrendTriggerFactor::new(0).expect("CudaTrendTriggerFactor::new");
    let result = cuda.batch_dev(&high, &low, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-10),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
