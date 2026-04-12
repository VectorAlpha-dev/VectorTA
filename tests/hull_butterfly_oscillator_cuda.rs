use vector_ta::indicators::hull_butterfly_oscillator::{
    hull_butterfly_oscillator_batch_with_kernel, HullButterflyOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaHullButterflyOscillator};

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
fn hull_butterfly_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[hull_butterfly_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2176usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.013).sin() * 0.41 + (x * 0.004).cos() * 0.17;
        data[i] = base + (x * 0.021).sin() * 0.73 + (x * 0.007).cos() * 0.22;
    }
    for i in (420..505).step_by(10) {
        data[i] = f64::NAN;
    }
    for i in (1320..1395).step_by(12) {
        data[i] = f64::NAN;
    }

    let sweep = HullButterflyOscillatorBatchRange {
        length: (10, 14, 2),
        mult: (1.5, 2.0, 0.5),
    };
    let cpu = hull_butterfly_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaHullButterflyOscillator::new(0).expect("CudaHullButterflyOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_oscillator = vec![0.0f64; result.outputs.oscillator.len()];
    let mut got_cumulative_mean = vec![0.0f64; result.outputs.cumulative_mean.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.oscillator.buf.copy_to(&mut got_oscillator)?;
    result
        .outputs
        .cumulative_mean
        .buf
        .copy_to(&mut got_cumulative_mean)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.oscillator.len() {
        assert!(
            approx_eq(cpu.oscillator[idx], got_oscillator[idx], 1e-9),
            "oscillator mismatch at {idx}: cpu={} cuda={}",
            cpu.oscillator[idx],
            got_oscillator[idx]
        );
        assert!(
            approx_eq(cpu.cumulative_mean[idx], got_cumulative_mean[idx], 1e-9),
            "cumulative_mean mismatch at {idx}: cpu={} cuda={}",
            cpu.cumulative_mean[idx],
            got_cumulative_mean[idx]
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
