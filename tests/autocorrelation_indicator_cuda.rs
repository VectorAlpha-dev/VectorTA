use vector_ta::indicators::autocorrelation_indicator::{
    autocorrelation_indicator_batch_with_kernel, AutocorrelationIndicatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAutocorrelationIndicator};

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
fn autocorrelation_indicator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[autocorrelation_indicator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1792usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 94.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.36 + (x * 0.004).cos() * 0.14;
        data[i] = base + (x * 0.023).sin() * 0.62 + (x * 0.007).cos() * 0.19;
    }
    for i in (420..500).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1180..1260).step_by(10) {
        data[i] = f64::NAN;
    }

    let sweep = AutocorrelationIndicatorBatchRange {
        length: (12, 20, 4),
        max_lag: Some(8),
        use_test_signal: Some(false),
    };
    let cpu = autocorrelation_indicator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAutocorrelationIndicator::new(0).expect("CudaAutocorrelationIndicator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.outputs.correlations.lag_count, cpu.lag_count);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_filtered = vec![0.0f64; result.outputs.filtered.len()];
    let mut got_correlations = vec![0.0f64; result.outputs.correlations.len()];
    result.outputs.filtered.buf.copy_to(&mut got_filtered)?;
    result
        .outputs
        .correlations
        .buf
        .copy_to(&mut got_correlations)?;

    for idx in 0..cpu.filtered.len() {
        assert!(
            approx_eq(cpu.filtered[idx], got_filtered[idx], 1e-8),
            "filtered mismatch at {idx}: cpu={} cuda={}",
            cpu.filtered[idx],
            got_filtered[idx]
        );
    }
    for idx in 0..cpu.correlations.len() {
        assert!(
            approx_eq(cpu.correlations[idx], got_correlations[idx], 1e-6),
            "correlations mismatch at {idx}: cpu={} cuda={}",
            cpu.correlations[idx],
            got_correlations[idx]
        );
    }

    Ok(())
}
