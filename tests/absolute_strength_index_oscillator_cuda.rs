use vector_ta::indicators::absolute_strength_index_oscillator::{
    absolute_strength_index_oscillator_batch_with_kernel, AbsoluteStrengthIndexOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAbsoluteStrengthIndexOscillator};

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
fn absolute_strength_index_oscillator_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[absolute_strength_index_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 6..len {
        let x = i as f64;
        value += (x * 0.019).sin() * 0.61 + (x * 0.007).cos() * 0.24;
        data[i] = value + (x * 0.013).sin() * 0.17;
    }
    for i in (420..500).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1320..1390).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = AbsoluteStrengthIndexOscillatorBatchRange {
        ema_length: (5, 7, 2),
        signal_length: (3, 5, 2),
    };
    let cpu =
        absolute_strength_index_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAbsoluteStrengthIndexOscillator::new(0)
        .expect("CudaAbsoluteStrengthIndexOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_oscillator = vec![0.0f64; result.outputs.oscillator.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    let mut got_histogram = vec![0.0f64; result.outputs.histogram.len()];
    result.outputs.oscillator.buf.copy_to(&mut got_oscillator)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;
    result.outputs.histogram.buf.copy_to(&mut got_histogram)?;

    for idx in 0..cpu.oscillator.len() {
        assert!(
            approx_eq(cpu.oscillator[idx], got_oscillator[idx], 1e-12),
            "oscillator mismatch at {idx}: cpu={} cuda={}",
            cpu.oscillator[idx],
            got_oscillator[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-12),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
        assert!(
            approx_eq(cpu.histogram[idx], got_histogram[idx], 1e-12),
            "histogram mismatch at {idx}: cpu={} cuda={}",
            cpu.histogram[idx],
            got_histogram[idx]
        );
    }

    Ok(())
}
