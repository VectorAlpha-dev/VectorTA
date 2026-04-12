use vector_ta::indicators::qqe_weighted_oscillator::{
    qqe_weighted_oscillator_batch_with_kernel, QqeWeightedOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaQqeWeightedOscillator};

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
fn qqe_weighted_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[qqe_weighted_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2112usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 10..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.41 + (x * 0.004).cos() * 0.23;
        data[i] = base + (x * 0.017).sin() * 0.79 + (x * 0.006).cos() * 0.28;
    }
    for i in (440..520).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1320..1400).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = QqeWeightedOscillatorBatchRange {
        length: (10, 14, 2),
        factor: (3.0, 4.0, 1.0),
        smooth: (4, 6, 2),
        weight: (1.5, 2.5, 1.0),
    };
    let cpu = qqe_weighted_oscillator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaQqeWeightedOscillator::new(0).expect("CudaQqeWeightedOscillator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_rsi = vec![0.0f64; result.outputs.rsi.len()];
    let mut got_ts = vec![0.0f64; result.outputs.trailing_stop.len()];
    result.outputs.rsi.buf.copy_to(&mut got_rsi)?;
    result.outputs.trailing_stop.buf.copy_to(&mut got_ts)?;

    for idx in 0..cpu.rsi.len() {
        assert!(
            approx_eq(cpu.rsi[idx], got_rsi[idx], 1e-6),
            "rsi mismatch at {idx}: cpu={} cuda={}",
            cpu.rsi[idx],
            got_rsi[idx]
        );
        assert!(
            approx_eq(cpu.trailing_stop[idx], got_ts[idx], 1e-6),
            "trailing_stop mismatch at {idx}: cpu={} cuda={}",
            cpu.trailing_stop[idx],
            got_ts[idx]
        );
    }

    Ok(())
}
