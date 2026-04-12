use vector_ta::indicators::adaptive_macd::{
    adaptive_macd_batch_with_kernel, AdaptiveMacdBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAdaptiveMacd};

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
fn adaptive_macd_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adaptive_macd_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1856usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 6..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.34 + (x * 0.004).cos() * 0.16;
        data[i] = base + (x * 0.014).sin() * 0.49 + (x * 0.007).cos() * 0.23;
    }
    for i in (430..500).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1180..1245).step_by(8) {
        data[i] = f64::NAN;
    }

    let sweep = AdaptiveMacdBatchRange {
        length: (5, 7, 2),
        fast_period: (4, 6, 2),
        slow_period: (8, 10, 2),
        signal_period: (3, 5, 2),
    };
    let cpu = adaptive_macd_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAdaptiveMacd::new(0).expect("CudaAdaptiveMacd::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_macd = vec![0.0f64; result.outputs.macd.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    let mut got_hist = vec![0.0f64; result.outputs.hist.len()];
    result.outputs.macd.buf.copy_to(&mut got_macd)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;
    result.outputs.hist.buf.copy_to(&mut got_hist)?;

    for idx in 0..cpu.macd.len() {
        assert!(
            approx_eq(cpu.macd[idx], got_macd[idx], 1e-6),
            "macd mismatch at {idx}: cpu={} cuda={}",
            cpu.macd[idx],
            got_macd[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-6),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
        assert!(
            approx_eq(cpu.hist[idx], got_hist[idx], 1e-6),
            "hist mismatch at {idx}: cpu={} cuda={}",
            cpu.hist[idx],
            got_hist[idx]
        );
    }

    Ok(())
}
