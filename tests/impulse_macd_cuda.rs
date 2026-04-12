use vector_ta::indicators::impulse_macd::{impulse_macd_batch_with_kernel, ImpulseMacdBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaImpulseMacd};

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
fn impulse_macd_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[impulse_macd_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2112usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 11..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.37 + (x * 0.004).cos() * 0.16;
        close[i] = base + (x * 0.018).sin() * 0.58;
        high[i] = close[i] + 0.81 + (x * 0.011).sin().abs() * 0.22;
        low[i] = close[i] - 0.77 - (x * 0.013).cos().abs() * 0.19;
    }
    for i in (430..510).step_by(9) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1360..1440).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = ImpulseMacdBatchRange {
        length_ma: (24, 34, 10),
        length_signal: (5, 9, 4),
    };
    let cpu = impulse_macd_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaImpulseMacd::new(0).expect("CudaImpulseMacd::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_md = vec![0.0f64; result.outputs.impulse_macd.len()];
    let mut got_hist = vec![0.0f64; result.outputs.impulse_histo.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.impulse_macd.buf.copy_to(&mut got_md)?;
    result.outputs.impulse_histo.buf.copy_to(&mut got_hist)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.impulse_macd.len() {
        assert!(
            approx_eq(cpu.impulse_macd[idx], got_md[idx], 1e-9),
            "impulse_macd mismatch at {idx}: cpu={} cuda={}",
            cpu.impulse_macd[idx],
            got_md[idx]
        );
        assert!(
            approx_eq(cpu.impulse_histo[idx], got_hist[idx], 1e-9),
            "impulse_histo mismatch at {idx}: cpu={} cuda={}",
            cpu.impulse_histo[idx],
            got_hist[idx]
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
