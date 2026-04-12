use vector_ta::indicators::projection_oscillator::{
    projection_oscillator_batch_with_kernel, ProjectionOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaProjectionOscillator};

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
fn projection_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[projection_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1920usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 105.0f64;
    for i in 6..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.63 + (x * 0.004).cos() * 0.22;
        let center = base + (x * 0.018).sin() * 0.27;
        high[i] = center + 0.85 + (x * 0.012).cos().abs() * 0.16;
        low[i] = center - 0.88 - (x * 0.009).sin().abs() * 0.19;
        close[i] = center + (x * 0.014).sin() * 0.21;
    }
    for i in (620..700).step_by(11) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1440..1510).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = ProjectionOscillatorBatchRange {
        length: (10, 14, 2),
        smooth_length: (3, 5, 2),
    };
    let cpu =
        projection_oscillator_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaProjectionOscillator::new(0).expect("CudaProjectionOscillator::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_pbo = vec![0.0f64; result.outputs.pbo.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.pbo.buf.copy_to(&mut got_pbo)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.pbo.len() {
        assert!(
            approx_eq(cpu.pbo[idx], got_pbo[idx], 1e-10),
            "pbo mismatch at {idx}: cpu={} cuda={}",
            cpu.pbo[idx],
            got_pbo[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-10),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
