use vector_ta::indicators::evasive_supertrend::{
    evasive_supertrend_batch_with_kernel, EvasiveSuperTrendBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEvasiveSuperTrend};

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
fn evasive_supertrend_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[evasive_supertrend_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 600usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 96.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.008).sin() * 0.24 + (x * 0.0023).cos() * 0.07;
        close[i] = base + (x * 0.043).sin() * 1.1 + (x * 0.019).cos() * 0.6;
        open[i] = close[i] - (x * 0.031).sin() * 0.45;
        let span = 1.05 + (x * 0.014).sin().abs() * 0.6;
        high[i] = close[i].max(open[i]) + span;
        low[i] = close[i].min(open[i]) - span * 0.9;
    }
    open[311] = f64::NAN;
    high[311] = f64::NAN;
    low[311] = f64::NAN;
    close[311] = f64::NAN;

    let sweep = EvasiveSuperTrendBatchRange {
        atr_length: (10, 12, 2),
        base_multiplier: (2.5, 3.0, 0.5),
        noise_threshold: (1.0, 1.5, 0.5),
        expansion_alpha: (0.5, 0.7, 0.2),
    };

    let cpu = evasive_supertrend_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaEvasiveSuperTrend::new(0).expect("CudaEvasiveSuperTrend::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_band = vec![0.0f64; result.outputs.band.len()];
    let mut got_state = vec![0.0f64; result.outputs.state.len()];
    let mut got_noisy = vec![0.0f64; result.outputs.noisy.len()];
    let mut got_changed = vec![0.0f64; result.outputs.changed.len()];
    result.outputs.band.buf.copy_to(&mut got_band)?;
    result.outputs.state.buf.copy_to(&mut got_state)?;
    result.outputs.noisy.buf.copy_to(&mut got_noisy)?;
    result.outputs.changed.buf.copy_to(&mut got_changed)?;

    for idx in 0..cpu.band.len() {
        assert!(
            approx_eq(cpu.band[idx], got_band[idx], 1e-6),
            "band mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.state[idx], got_state[idx], 1e-6),
            "state mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.noisy[idx], got_noisy[idx], 1e-6),
            "noisy mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.changed[idx], got_changed[idx], 1e-6),
            "changed mismatch at {idx}"
        );
    }

    Ok(())
}
