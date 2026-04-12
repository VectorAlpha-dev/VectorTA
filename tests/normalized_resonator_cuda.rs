use vector_ta::indicators::normalized_resonator::{
    normalized_resonator_batch_with_kernel, NormalizedResonatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaNormalizedResonator};

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
fn normalized_resonator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[normalized_resonator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2304usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 118.0f64;
    for i in 16..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.52 + (x * 0.003).cos() * 0.18;
        data[i] = base + (x * 0.021).sin() * 0.81 + (x * 0.005).cos() * 0.27;
    }
    for i in (580..650).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1510..1580).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = NormalizedResonatorBatchRange {
        period: (60, 100, 20),
        delta: (0.4, 0.6, 0.2),
        lookback_mult: (1.0, 2.0, 1.0),
        signal_length: (5, 9, 4),
    };
    let cpu = normalized_resonator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaNormalizedResonator::new(0).expect("CudaNormalizedResonator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_oscillator = vec![0.0f64; result.outputs.oscillator.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.oscillator.buf.copy_to(&mut got_oscillator)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.oscillator.len() {
        assert!(
            approx_eq(cpu.oscillator[idx], got_oscillator[idx], 1e-6),
            "oscillator mismatch at {idx}: cpu={} cuda={}",
            cpu.oscillator[idx],
            got_oscillator[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-6),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
