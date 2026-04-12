use vector_ta::indicators::polynomial_regression_extrapolation::{
    polynomial_regression_extrapolation_batch_with_kernel,
    PolynomialRegressionExtrapolationBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaPolynomialRegressionExtrapolation};

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
fn polynomial_regression_extrapolation_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[polynomial_regression_extrapolation_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 1728usize;
    let mut data = vec![f64::NAN; len];
    for (idx, value) in data.iter_mut().enumerate().skip(9) {
        let x = idx as f64 * 0.09;
        *value = 18.0 + 0.35 * x * x - 0.002 * x * x * x + (x * 0.41).sin() * 0.7;
    }
    for idx in (480..540).step_by(12) {
        data[idx] = f64::NAN;
    }
    for idx in (1180..1240).step_by(11) {
        data[idx] = f64::NAN;
    }

    let sweep = PolynomialRegressionExtrapolationBatchRange {
        length: (5, 7, 2),
        extrapolate: (1, 2, 1),
        degree: (1, 3, 1),
    };
    let cpu =
        polynomial_regression_extrapolation_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaPolynomialRegressionExtrapolation::new(0)
        .expect("CudaPolynomialRegressionExtrapolation::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-10),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
