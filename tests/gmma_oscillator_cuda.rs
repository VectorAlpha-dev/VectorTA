use vector_ta::indicators::gmma_oscillator::{
    gmma_oscillator_batch_with_kernel, GmmaOscillatorBatchRange, GmmaOscillatorParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaGmmaOscillator};

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
fn gmma_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[gmma_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2304usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 116.0f64;
    for i in 16..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.39 + (x * 0.005).cos() * 0.15;
        data[i] = base + (x * 0.019).sin() * 0.58 + (x * 0.006).cos() * 0.21;
    }
    for i in (510..590).step_by(11) {
        data[i] = f64::NAN;
    }
    for i in (1490..1570).step_by(13) {
        data[i] = f64::NAN;
    }

    let sweep = GmmaOscillatorBatchRange {
        smooth_length: (1, 3, 2),
        signal_length: (7, 11, 4),
    };
    let fixed = GmmaOscillatorParams {
        gmma_type: Some("super_guppy".to_string()),
        smooth_length: None,
        signal_length: None,
        anchor_minutes: Some(60),
        interval_minutes: Some(15),
    };
    let cpu = gmma_oscillator_batch_with_kernel(&data, &sweep, &fixed, Kernel::ScalarBatch)?;
    let cuda = CudaGmmaOscillator::new(0).expect("CudaGmmaOscillator::new");
    let result = cuda.batch_dev(&data, &sweep, &fixed).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_oscillator = vec![0.0f64; result.outputs.oscillator.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.oscillator.buf.copy_to(&mut got_oscillator)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.oscillator.len() {
        assert!(
            approx_eq(cpu.oscillator[idx], got_oscillator[idx], 1e-9),
            "oscillator mismatch at {idx}: cpu={} cuda={}",
            cpu.oscillator[idx],
            got_oscillator[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-9),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
