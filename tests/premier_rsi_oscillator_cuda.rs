use vector_ta::indicators::premier_rsi_oscillator::{
    premier_rsi_oscillator_batch_with_kernel, PremierRsiOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaPremierRsiOscillator};

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
fn premier_rsi_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[premier_rsi_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1952usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 92.0f64;
    for i in 8..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.47 + (x * 0.005).cos() * 0.19;
        data[i] = base + (x * 0.018).sin() * 0.86 + (x * 0.007).cos() * 0.31;
    }
    for i in (430..500).step_by(11) {
        data[i] = f64::NAN;
    }
    for i in (1260..1340).step_by(13) {
        data[i] = f64::NAN;
    }

    let sweep = PremierRsiOscillatorBatchRange {
        rsi_length: (10, 14, 2),
        stoch_length: (6, 8, 2),
        smooth_length: (16, 20, 4),
    };
    let cpu = premier_rsi_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaPremierRsiOscillator::new(0).expect("CudaPremierRsiOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_values = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got_values)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got_values[idx], 1e-6),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got_values[idx]
        );
    }

    Ok(())
}
