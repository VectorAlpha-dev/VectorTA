use vector_ta::indicators::supertrend_oscillator::{
    supertrend_oscillator_batch_with_kernel, SuperTrendOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaSupertrendOscillator};

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
fn supertrend_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[supertrend_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2240usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut source = vec![f64::NAN; len];
    let mut base = 99.0f64;
    for i in 14..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.28 + (x * 0.004).cos() * 0.15;
        source[i] = base + (x * 0.018).sin() * 0.54 + (x * 0.006).cos() * 0.19;
        high[i] = source[i] + 0.84 + (x * 0.013).sin().abs() * 0.22;
        low[i] = source[i] - 0.82 - (x * 0.012).cos().abs() * 0.20;
    }
    for i in (420..520).step_by(11) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        source[i] = f64::NAN;
    }
    for i in (1360..1450).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        source[i] = f64::NAN;
    }

    let sweep = SuperTrendOscillatorBatchRange {
        length: (10, 12, 2),
        mult: (1.5, 2.5, 1.0),
        smooth: (5, 7, 2),
    };
    let cpu =
        supertrend_oscillator_batch_with_kernel(&high, &low, &source, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaSupertrendOscillator::new(0).expect("CudaSupertrendOscillator::new");
    let result = cuda
        .batch_dev(&high, &low, &source, &sweep)
        .expect("batch_dev");

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
        assert!(
            approx_eq(cpu.histogram[idx], got_histogram[idx], 1e-6),
            "histogram mismatch at {idx}: cpu={} cuda={}",
            cpu.histogram[idx],
            got_histogram[idx]
        );
    }

    Ok(())
}
