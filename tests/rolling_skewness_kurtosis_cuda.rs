use vector_ta::indicators::rolling_skewness_kurtosis::{
    rolling_skewness_kurtosis_batch_with_kernel, RollingSkewnessKurtosisBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRollingSkewnessKurtosis};

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
fn rolling_skewness_kurtosis_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[rolling_skewness_kurtosis_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 95.0f64;
    for i in 6..len {
        let x = i as f64;
        value += (x * 0.013).sin() * 0.66 + (x * 0.004).cos() * 0.24;
        data[i] = value + (x * 0.027).sin() * 0.31;
    }
    for i in (700..770).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1500..1580).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = RollingSkewnessKurtosisBatchRange {
        length: (32, 40, 8),
        smooth_length: (3, 5, 2),
    };
    let cpu = rolling_skewness_kurtosis_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaRollingSkewnessKurtosis::new(0).expect("CudaRollingSkewnessKurtosis::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_skewness = vec![0.0f64; result.outputs.skewness.len()];
    let mut got_kurtosis = vec![0.0f64; result.outputs.kurtosis.len()];
    result.outputs.skewness.buf.copy_to(&mut got_skewness)?;
    result.outputs.kurtosis.buf.copy_to(&mut got_kurtosis)?;

    for idx in 0..cpu.skewness.len() {
        assert!(
            approx_eq(cpu.skewness[idx], got_skewness[idx], 1e-8),
            "skewness mismatch at {idx}: cpu={} cuda={}",
            cpu.skewness[idx],
            got_skewness[idx]
        );
        assert!(
            approx_eq(cpu.kurtosis[idx], got_kurtosis[idx], 1e-8),
            "kurtosis mismatch at {idx}: cpu={} cuda={}",
            cpu.kurtosis[idx],
            got_kurtosis[idx]
        );
    }

    Ok(())
}
