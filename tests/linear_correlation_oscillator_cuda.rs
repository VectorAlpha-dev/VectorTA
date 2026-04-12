use vector_ta::indicators::linear_correlation_oscillator::{
    linear_correlation_oscillator_batch_with_kernel, LinearCorrelationOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaLinearCorrelationOscillator};

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
fn linear_correlation_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[linear_correlation_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 90.0f64;
    for i in 5..len {
        let x = i as f64;
        value += (x * 0.014).sin() * 0.55 + (x * 0.006).cos() * 0.24;
        data[i] = value + (x * 0.019).sin() * 0.17;
    }
    for i in (540..610).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1320..1385).step_by(8) {
        data[i] = f64::NAN;
    }

    let sweep = LinearCorrelationOscillatorBatchRange {
        period: (12, 18, 3),
    };
    let cpu = linear_correlation_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda =
        CudaLinearCorrelationOscillator::new(0).expect("CudaLinearCorrelationOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-9),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
