#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaLinearRegressionIntensity};
use vector_ta::indicators::linear_regression_intensity::{
    linear_regression_intensity, LinearRegressionIntensityBatchRange,
    LinearRegressionIntensityInput,
};

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
fn linear_regression_intensity_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[linear_regression_intensity_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2304usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 80.0f64;
    for i in 7..len {
        let x = i as f64;
        base += (x * 0.008).sin() * 0.42 + (x * 0.002).cos() * 0.17;
        data[i] = base + (x * 0.019).sin() * 1.7 + (x * 0.007).cos() * 0.55;
    }

    let sweep = LinearRegressionIntensityBatchRange {
        lookback_period: (10, 14, 2),
        range_tolerance: (85.0, 95.0, 10.0),
        linreg_length: (20, 28, 4),
    };
    let cuda = CudaLinearRegressionIntensity::new(0).expect("CudaLinearRegressionIntensity::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, result.combos.len());
    assert_eq!(result.outputs.cols, data.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for (row, combo) in result.combos.iter().enumerate() {
        let cpu = linear_regression_intensity(&LinearRegressionIntensityInput::from_slice(
            &data,
            combo.clone(),
        ))?;
        let start = row * result.outputs.cols;
        for idx in 0..result.outputs.cols {
            assert!(
                approx_eq(cpu.values[idx], got[start + idx], 1e-10),
                "values mismatch at row={row} idx={idx}: cpu={} cuda={}",
                cpu.values[idx],
                got[start + idx]
            );
        }
    }

    Ok(())
}
