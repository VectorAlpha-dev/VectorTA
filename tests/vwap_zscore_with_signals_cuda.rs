use vector_ta::indicators::vwap_zscore_with_signals::{
    vwap_zscore_with_signals_batch_with_kernel, VwapZscoreWithSignalsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVwapZscoreWithSignals};

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
fn vwap_zscore_with_signals_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[vwap_zscore_with_signals_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2240usize;
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut base = 109.0f64;
    for i in 14..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.42 + (x * 0.003).cos() * 0.18;
        close[i] = base + (x * 0.017).sin() * 0.61;
        volume[i] = 24_000.0 + (x * 0.014).sin() * 2_300.0 + (x % 19.0) * 97.0;
    }

    let sweep = VwapZscoreWithSignalsBatchRange {
        length: (18, 20, 2),
        upper_bottom: (2.0, 2.5, 0.5),
        lower_bottom: (-2.5, -2.0, 0.5),
    };
    let cpu =
        vwap_zscore_with_signals_batch_with_kernel(&close, &volume, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaVwapZscoreWithSignals::new(0).expect("CudaVwapZscoreWithSignals::new");
    let result = cuda.batch_dev(&close, &volume, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_zvwap = vec![0.0f64; result.outputs.zvwap.len()];
    let mut got_support = vec![0.0f64; result.outputs.support_signal.len()];
    let mut got_resistance = vec![0.0f64; result.outputs.resistance_signal.len()];
    result.outputs.zvwap.buf.copy_to(&mut got_zvwap)?;
    result
        .outputs
        .support_signal
        .buf
        .copy_to(&mut got_support)?;
    result
        .outputs
        .resistance_signal
        .buf
        .copy_to(&mut got_resistance)?;

    for idx in 0..cpu.zvwap.len() {
        assert!(
            approx_eq(cpu.zvwap[idx], got_zvwap[idx], 1e-9),
            "zvwap mismatch at {idx}: cpu={} cuda={}",
            cpu.zvwap[idx],
            got_zvwap[idx]
        );
        assert!(
            approx_eq(cpu.support_signal[idx], got_support[idx], 1e-9),
            "support mismatch at {idx}: cpu={} cuda={}",
            cpu.support_signal[idx],
            got_support[idx]
        );
        assert!(
            approx_eq(cpu.resistance_signal[idx], got_resistance[idx], 1e-9),
            "resistance mismatch at {idx}: cpu={} cuda={}",
            cpu.resistance_signal[idx],
            got_resistance[idx]
        );
    }

    Ok(())
}
