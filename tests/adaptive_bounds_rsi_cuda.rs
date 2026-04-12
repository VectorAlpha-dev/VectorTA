use vector_ta::indicators::adaptive_bounds_rsi::{
    adaptive_bounds_rsi_batch_with_kernel, AdaptiveBoundsRsiBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAdaptiveBoundsRsi};

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
fn adaptive_bounds_rsi_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adaptive_bounds_rsi_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2176usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.013).sin() * 0.44 + (x * 0.003).cos() * 0.18;
        data[i] = base + (x * 0.017).sin() * 0.82 + (x * 0.005).cos() * 0.31;
    }
    for i in (420..500).step_by(11) {
        data[i] = f64::NAN;
    }
    for i in (1280..1360).step_by(13) {
        data[i] = f64::NAN;
    }

    let sweep = AdaptiveBoundsRsiBatchRange {
        rsi_length: (10, 14, 2),
        alpha: (0.08, 0.12, 0.02),
    };
    let cpu = adaptive_bounds_rsi_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAdaptiveBoundsRsi::new(0).expect("CudaAdaptiveBoundsRsi::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_rsi = vec![0.0f64; result.outputs.rsi.len()];
    let mut got_lower_bound = vec![0.0f64; result.outputs.lower_bound.len()];
    let mut got_lower_mid = vec![0.0f64; result.outputs.lower_mid.len()];
    let mut got_mid = vec![0.0f64; result.outputs.mid.len()];
    let mut got_upper_mid = vec![0.0f64; result.outputs.upper_mid.len()];
    let mut got_upper_bound = vec![0.0f64; result.outputs.upper_bound.len()];
    let mut got_regime = vec![0.0f64; result.outputs.regime.len()];
    let mut got_regime_flip = vec![0.0f64; result.outputs.regime_flip.len()];
    let mut got_lower_signal = vec![0.0f64; result.outputs.lower_signal.len()];
    let mut got_upper_signal = vec![0.0f64; result.outputs.upper_signal.len()];
    result.outputs.rsi.buf.copy_to(&mut got_rsi)?;
    result
        .outputs
        .lower_bound
        .buf
        .copy_to(&mut got_lower_bound)?;
    result.outputs.lower_mid.buf.copy_to(&mut got_lower_mid)?;
    result.outputs.mid.buf.copy_to(&mut got_mid)?;
    result.outputs.upper_mid.buf.copy_to(&mut got_upper_mid)?;
    result
        .outputs
        .upper_bound
        .buf
        .copy_to(&mut got_upper_bound)?;
    result.outputs.regime.buf.copy_to(&mut got_regime)?;
    result
        .outputs
        .regime_flip
        .buf
        .copy_to(&mut got_regime_flip)?;
    result
        .outputs
        .lower_signal
        .buf
        .copy_to(&mut got_lower_signal)?;
    result
        .outputs
        .upper_signal
        .buf
        .copy_to(&mut got_upper_signal)?;

    for idx in 0..cpu.rsi.len() {
        assert!(
            approx_eq(cpu.rsi[idx], got_rsi[idx], 1e-9),
            "rsi mismatch at {idx}: cpu={} cuda={}",
            cpu.rsi[idx],
            got_rsi[idx]
        );
        assert!(
            approx_eq(cpu.lower_bound[idx], got_lower_bound[idx], 1e-9),
            "lower_bound mismatch at {idx}: cpu={} cuda={}",
            cpu.lower_bound[idx],
            got_lower_bound[idx]
        );
        assert!(
            approx_eq(cpu.lower_mid[idx], got_lower_mid[idx], 1e-9),
            "lower_mid mismatch at {idx}: cpu={} cuda={}",
            cpu.lower_mid[idx],
            got_lower_mid[idx]
        );
        assert!(
            approx_eq(cpu.mid[idx], got_mid[idx], 1e-9),
            "mid mismatch at {idx}: cpu={} cuda={}",
            cpu.mid[idx],
            got_mid[idx]
        );
        assert!(
            approx_eq(cpu.upper_mid[idx], got_upper_mid[idx], 1e-9),
            "upper_mid mismatch at {idx}: cpu={} cuda={}",
            cpu.upper_mid[idx],
            got_upper_mid[idx]
        );
        assert!(
            approx_eq(cpu.upper_bound[idx], got_upper_bound[idx], 1e-9),
            "upper_bound mismatch at {idx}: cpu={} cuda={}",
            cpu.upper_bound[idx],
            got_upper_bound[idx]
        );
        assert!(
            approx_eq(cpu.regime[idx], got_regime[idx], 1e-12),
            "regime mismatch at {idx}: cpu={} cuda={}",
            cpu.regime[idx],
            got_regime[idx]
        );
        assert!(
            approx_eq(cpu.regime_flip[idx], got_regime_flip[idx], 1e-12),
            "regime_flip mismatch at {idx}: cpu={} cuda={}",
            cpu.regime_flip[idx],
            got_regime_flip[idx]
        );
        assert!(
            approx_eq(cpu.lower_signal[idx], got_lower_signal[idx], 1e-12),
            "lower_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.lower_signal[idx],
            got_lower_signal[idx]
        );
        assert!(
            approx_eq(cpu.upper_signal[idx], got_upper_signal[idx], 1e-12),
            "upper_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.upper_signal[idx],
            got_upper_signal[idx]
        );
    }

    Ok(())
}
