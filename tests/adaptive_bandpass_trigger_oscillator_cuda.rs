use vector_ta::indicators::adaptive_bandpass_trigger_oscillator::{
    adaptive_bandpass_trigger_oscillator_batch_with_kernel,
    AdaptiveBandpassTriggerOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAdaptiveBandpassTriggerOscillator};

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
fn adaptive_bandpass_trigger_oscillator_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[adaptive_bandpass_trigger_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2176usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 106.0f64;
    for i in 14..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.47 + (x * 0.003).cos() * 0.14;
        data[i] = base + (x * 0.019).sin() * 0.74 + (x * 0.007).cos() * 0.21;
    }
    for i in (500..570).step_by(8) {
        data[i] = f64::NAN;
    }
    for i in (1390..1460).step_by(10) {
        data[i] = f64::NAN;
    }

    let sweep = AdaptiveBandpassTriggerOscillatorBatchRange {
        delta: (0.08, 0.12, 0.02),
        alpha: (0.05, 0.09, 0.02),
    };
    let cpu =
        adaptive_bandpass_trigger_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAdaptiveBandpassTriggerOscillator::new(0)
        .expect("CudaAdaptiveBandpassTriggerOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_in_phase = vec![0.0f64; result.outputs.in_phase.len()];
    let mut got_lead = vec![0.0f64; result.outputs.lead.len()];
    result.outputs.in_phase.buf.copy_to(&mut got_in_phase)?;
    result.outputs.lead.buf.copy_to(&mut got_lead)?;

    for idx in 0..cpu.in_phase.len() {
        assert!(
            approx_eq(cpu.in_phase[idx], got_in_phase[idx], 1e-6),
            "in_phase mismatch at {idx}: cpu={} cuda={}",
            cpu.in_phase[idx],
            got_in_phase[idx]
        );
        assert!(
            approx_eq(cpu.lead[idx], got_lead[idx], 1e-6),
            "lead mismatch at {idx}: cpu={} cuda={}",
            cpu.lead[idx],
            got_lead[idx]
        );
    }

    Ok(())
}
