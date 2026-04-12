use vector_ta::indicators::momentum_ratio_oscillator::{
    momentum_ratio_oscillator_batch_with_kernel, MomentumRatioOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMomentumRatioOscillator};

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
fn momentum_ratio_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[momentum_ratio_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 8..len {
        value += (i as f64 * 0.015).sin() * 0.8 + (i as f64 * 0.007).cos() * 0.35;
        data[i] = value;
    }
    for i in (1000..1080).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = MomentumRatioOscillatorBatchRange {
        period: (20, 60, 20),
    };
    let cpu = momentum_ratio_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaMomentumRatioOscillator::new(0).expect("CudaMomentumRatioOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_line = vec![0f64; result.outputs.line.len()];
    let mut got_signal = vec![0f64; result.outputs.signal.len()];
    result.outputs.line.buf.copy_to(&mut got_line)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.line.len() {
        assert!(
            approx_eq(cpu.line[idx], got_line[idx], 1e-8),
            "line mismatch at {idx}: cpu={} cuda={}",
            cpu.line[idx],
            got_line[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-8),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
