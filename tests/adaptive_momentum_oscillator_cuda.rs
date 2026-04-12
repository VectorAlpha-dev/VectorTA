use vector_ta::indicators::adaptive_momentum_oscillator::{
    adaptive_momentum_oscillator_batch_with_kernel, AdaptiveMomentumOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAdaptiveMomentumOscillator};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        let scale = a.abs().max(b.abs()).max(1.0);
        (a - b).abs() <= tol.max(scale * 1e-12)
    }
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn adaptive_momentum_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adaptive_momentum_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1024usize;
    let mut data = vec![f64::NAN; len];
    for i in 7..len {
        let x = i as f64;
        data[i] =
            100.0 + (x * 0.091).sin() * 3.4 + (x * 0.037).cos() * 1.6 + (x * 0.013).sin() * 0.75;
    }
    for i in (280..340).step_by(10) {
        data[i] = f64::NAN;
    }
    for i in (700..760).step_by(9) {
        data[i] = f64::NAN;
    }

    let sweep = AdaptiveMomentumOscillatorBatchRange {
        length: (10, 14, 4),
        smoothing_length: (4, 6, 2),
    };
    let cpu = adaptive_momentum_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAdaptiveMomentumOscillator::new(0).expect("CudaAdaptiveMomentumOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_amo = vec![0.0f64; result.outputs.amo.len()];
    let mut got_ama = vec![0.0f64; result.outputs.ama.len()];
    result.outputs.amo.buf.copy_to(&mut got_amo)?;
    result.outputs.ama.buf.copy_to(&mut got_ama)?;

    for idx in 0..cpu.amo.len() {
        assert!(
            approx_eq(cpu.amo[idx], got_amo[idx], 1e-9),
            "amo mismatch at {idx}: cpu={} cuda={}",
            cpu.amo[idx],
            got_amo[idx]
        );
        assert!(
            approx_eq(cpu.ama[idx], got_ama[idx], 1e-9),
            "ama mismatch at {idx}: cpu={} cuda={}",
            cpu.ama[idx],
            got_ama[idx]
        );
    }

    Ok(())
}
