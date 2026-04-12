use vector_ta::indicators::spearman_correlation::{
    spearman_correlation_batch_with_kernel, SpearmanCorrelationBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaSpearmanCorrelation};

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
fn spearman_correlation_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[spearman_correlation_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1664usize;
    let mut main = vec![f64::NAN; len];
    let mut compare = vec![f64::NAN; len];
    let mut base_main = 51.0f64;
    let mut base_compare = 77.0f64;
    for i in 6..len {
        let x = i as f64;
        base_main += (x * 0.012).sin() * 0.36 + (x * 0.004).cos() * 0.14;
        base_compare += (x * 0.010).cos() * 0.31 + (x * 0.005).sin() * 0.17;
        main[i] = base_main + (x * 0.017).sin() * 0.43 + (x * 0.003).cos() * 0.21;
        compare[i] = base_compare + (x * 0.015).sin() * 0.29 - (x * 0.007).cos() * 0.18;
    }
    for i in (520..600).step_by(10) {
        main[i] = f64::NAN;
        compare[i] = f64::NAN;
    }
    for i in (1210..1280).step_by(9) {
        main[i] = f64::NAN;
        compare[i] = f64::NAN;
    }

    let sweep = SpearmanCorrelationBatchRange {
        lookback: (8, 12, 2),
        smoothing_length: (3, 5, 2),
    };
    let cpu = spearman_correlation_batch_with_kernel(&main, &compare, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaSpearmanCorrelation::new(0).expect("CudaSpearmanCorrelation::new");
    let result = cuda.batch_dev(&main, &compare, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_raw = vec![0.0f64; result.outputs.raw.len()];
    let mut got_smoothed = vec![0.0f64; result.outputs.smoothed.len()];
    result.outputs.raw.buf.copy_to(&mut got_raw)?;
    result.outputs.smoothed.buf.copy_to(&mut got_smoothed)?;

    for idx in 0..cpu.raw.len() {
        assert!(
            approx_eq(cpu.raw[idx], got_raw[idx], 1e-10),
            "raw mismatch at {idx}: cpu={} cuda={}",
            cpu.raw[idx],
            got_raw[idx]
        );
        assert!(
            approx_eq(cpu.smoothed[idx], got_smoothed[idx], 1e-10),
            "smoothed mismatch at {idx}: cpu={} cuda={}",
            cpu.smoothed[idx],
            got_smoothed[idx]
        );
    }

    Ok(())
}
