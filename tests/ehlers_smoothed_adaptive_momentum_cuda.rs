use vector_ta::indicators::ehlers_smoothed_adaptive_momentum::{
    ehlers_smoothed_adaptive_momentum_batch_with_kernel, EhlersSmoothedAdaptiveMomentumBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersSmoothedAdaptiveMomentum};

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
fn ehlers_smoothed_adaptive_momentum_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[ehlers_smoothed_adaptive_momentum_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2304usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 94.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.014).sin() * 0.43 + (x * 0.004).cos() * 0.16;
        data[i] = base + (x * 0.022).sin() * 0.67 + (x * 0.006).cos() * 0.19;
    }
    for i in (540..620).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1510..1585).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = EhlersSmoothedAdaptiveMomentumBatchRange {
        alpha: (0.05, 0.09, 0.02),
        cutoff: (6.0, 10.0, 2.0),
    };
    let cpu =
        ehlers_smoothed_adaptive_momentum_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaEhlersSmoothedAdaptiveMomentum::new(0)
        .expect("CudaEhlersSmoothedAdaptiveMomentum::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-6),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
