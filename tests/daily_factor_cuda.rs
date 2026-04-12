use vector_ta::indicators::daily_factor::{daily_factor_batch_with_kernel, DailyFactorBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDailyFactor};

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
fn daily_factor_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[daily_factor_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1536usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 90.0f64;
    for i in 5..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.71 + (x * 0.004).cos() * 0.22;
        let spread = 0.8 + (x * 0.009).sin().abs() * 0.5;
        open[i] = base - 0.2 + (x * 0.006).cos() * 0.14;
        close[i] = base + 0.22 + (x * 0.008).sin() * 0.17;
        high[i] = open[i].max(close[i]) + spread * 0.55;
        low[i] = open[i].min(close[i]) - spread * 0.45;
    }
    for i in (380..450).step_by(13) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (980..1040).step_by(9) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = DailyFactorBatchRange {
        threshold_level: (0.25, 0.45, 0.10),
    };
    let cpu =
        daily_factor_batch_with_kernel(&open, &high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaDailyFactor::new(0).expect("CudaDailyFactor::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_value = vec![0.0f64; result.outputs.value.len()];
    let mut got_ema = vec![0.0f64; result.outputs.ema.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.value.buf.copy_to(&mut got_value)?;
    result.outputs.ema.buf.copy_to(&mut got_ema)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.value.len() {
        assert!(
            approx_eq(cpu.value[idx], got_value[idx], 1e-12),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.value[idx],
            got_value[idx]
        );
        assert!(
            approx_eq(cpu.ema[idx], got_ema[idx], 1e-12),
            "ema mismatch at {idx}: cpu={} cuda={}",
            cpu.ema[idx],
            got_ema[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-12),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
