use vector_ta::indicators::geometric_bias_oscillator::{
    geometric_bias_oscillator_batch_with_kernel, GeometricBiasOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaGeometricBiasOscillator};

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
fn geometric_bias_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[geometric_bias_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1728usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 132.0f64;
    for i in 8..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.41 + (x * 0.003).cos() * 0.19;
        let c = base + (x * 0.016).sin() * 0.58 + (x * 0.008).cos() * 0.21;
        close[i] = c;
        high[i] = c + 0.96 + (x * 0.013).cos().abs() * 0.17;
        low[i] = c - 0.92 - (x * 0.011).sin().abs() * 0.19;
    }
    for i in (540..620).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1290..1370).step_by(11) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = GeometricBiasOscillatorBatchRange {
        length: (20, 24, 4),
        multiplier: (1.5, 2.0, 0.5),
        atr_length: (10, 14, 4),
        smooth: (2, 3, 1),
    };
    let cpu = geometric_bias_oscillator_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaGeometricBiasOscillator::new(0).expect("CudaGeometricBiasOscillator::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_values = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got_values)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got_values[idx], 1e-10),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got_values[idx]
        );
    }

    Ok(())
}
