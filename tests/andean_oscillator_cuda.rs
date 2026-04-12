use vector_ta::indicators::andean_oscillator::{
    andean_oscillator_batch_with_kernel, AndeanOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAndeanOscillator};

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
fn andean_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[andean_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2144usize;
    let mut open = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 120.0f64;
    for i in 10..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.38 + (x * 0.004).cos() * 0.16;
        open[i] = base + (x * 0.007).sin() * 0.21;
        close[i] = open[i] + (x * 0.015).cos() * 0.94 + (x * 0.011).sin() * 0.27;
    }
    for i in (430..520).step_by(11) {
        open[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1360..1440).step_by(13) {
        open[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = AndeanOscillatorBatchRange {
        length: (30, 34, 2),
        signal_length: (7, 9, 2),
    };
    let cpu = andean_oscillator_batch_with_kernel(&open, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAndeanOscillator::new(0).expect("CudaAndeanOscillator::new");
    let result = cuda.batch_dev(&open, &close, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_bull = vec![0.0f64; result.outputs.bull.len()];
    let mut got_bear = vec![0.0f64; result.outputs.bear.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.bull.buf.copy_to(&mut got_bull)?;
    result.outputs.bear.buf.copy_to(&mut got_bear)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.bull.len() {
        assert!(
            approx_eq(cpu.bull[idx], got_bull[idx], 1e-5),
            "bull mismatch at {idx}: cpu={} cuda={}",
            cpu.bull[idx],
            got_bull[idx]
        );
        assert!(
            approx_eq(cpu.bear[idx], got_bear[idx], 1e-5),
            "bear mismatch at {idx}: cpu={} cuda={}",
            cpu.bear[idx],
            got_bear[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-5),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
