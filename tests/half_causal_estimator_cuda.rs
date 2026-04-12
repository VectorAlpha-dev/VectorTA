use vector_ta::indicators::half_causal_estimator::{
    half_causal_estimator_batch_with_kernel, HalfCausalEstimatorBatchRange,
    HalfCausalEstimatorConfidenceAdjust, HalfCausalEstimatorKernelType,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaHalfCausalEstimator};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_source(length: usize, slots_per_day: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(length);
    for i in 0..length {
        let slot = (i % slots_per_day) as f64;
        let day = (i / slots_per_day) as f64;
        out.push(
            1000.0
                + day * 5.0
                + (slot * 0.11).sin() * 30.0
                + (slot * 0.03).cos() * 12.0
                + (slot / slots_per_day as f64) * 25.0,
        );
    }
    out
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn half_causal_estimator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[half_causal_estimator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let slots_per_day = 48usize;
    let data = sample_source(slots_per_day * 6, slots_per_day);
    let sweep = HalfCausalEstimatorBatchRange {
        slots_per_day: Some(slots_per_day),
        data_period: (0, 5, 5),
        filter_length: (8, 8, 0),
        kernel_width: (10.0, 10.0, 0.0),
        maximum_confidence_adjust: (75.0, 75.0, 0.0),
        extra_smoothing: (0, 2, 2),
        kernel_type: HalfCausalEstimatorKernelType::Sinc,
        confidence_adjust: HalfCausalEstimatorConfidenceAdjust::Linear,
        enable_expected_value: true,
    };

    let cpu = half_causal_estimator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaHalfCausalEstimator::new(0)?;
    let result = cuda.batch_dev(&data, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_estimate = vec![0.0f64; result.outputs.estimate.len()];
    let mut got_expected_value = vec![0.0f64; result.outputs.expected_value.len()];
    result.outputs.estimate.buf.copy_to(&mut got_estimate)?;
    result
        .outputs
        .expected_value
        .buf
        .copy_to(&mut got_expected_value)?;

    for idx in 0..cpu.estimate_values.len() {
        assert!(
            approx_eq(cpu.estimate_values[idx], got_estimate[idx], 1e-6),
            "estimate mismatch at {idx}: cpu={} cuda={}",
            cpu.estimate_values[idx],
            got_estimate[idx]
        );
        assert!(
            approx_eq(cpu.expected_value_values[idx], got_expected_value[idx], 1e-6),
            "expected_value mismatch at {idx}: cpu={} cuda={}",
            cpu.expected_value_values[idx],
            got_expected_value[idx]
        );
    }

    Ok(())
}
