use vector_ta::indicators::supertrend_recovery::{
    supertrend_recovery_batch_with_kernel, SuperTrendRecoveryBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaSuperTrendRecovery};

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
fn supertrend_recovery_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[supertrend_recovery_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2304usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.31 + (x * 0.004).cos() * 0.18;
        close[i] = base + (x * 0.017).sin() * 0.58 + (x * 0.006).cos() * 0.22;
        high[i] = close[i] + 0.91 + (x * 0.013).sin().abs() * 0.25;
        low[i] = close[i] - 0.87 - (x * 0.011).cos().abs() * 0.22;
    }
    for i in (420..520).step_by(11) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1410..1500).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = SuperTrendRecoveryBatchRange {
        atr_length: (6, 8, 2),
        multiplier: (1.5, 2.5, 1.0),
        alpha_percent: (5.0, 10.0, 5.0),
        threshold_atr: (0.5, 1.0, 0.5),
    };
    let cpu =
        supertrend_recovery_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaSuperTrendRecovery::new(0).expect("CudaSuperTrendRecovery::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_band = vec![0.0f64; result.outputs.band.len()];
    let mut got_switch_price = vec![0.0f64; result.outputs.switch_price.len()];
    let mut got_trend = vec![0.0f64; result.outputs.trend.len()];
    let mut got_changed = vec![0.0f64; result.outputs.changed.len()];
    result.outputs.band.buf.copy_to(&mut got_band)?;
    result
        .outputs
        .switch_price
        .buf
        .copy_to(&mut got_switch_price)?;
    result.outputs.trend.buf.copy_to(&mut got_trend)?;
    result.outputs.changed.buf.copy_to(&mut got_changed)?;

    for idx in 0..cpu.band.len() {
        assert!(
            approx_eq(cpu.band[idx], got_band[idx], 1e-6),
            "band mismatch at {idx}: cpu={} cuda={}",
            cpu.band[idx],
            got_band[idx]
        );
        assert!(
            approx_eq(cpu.switch_price[idx], got_switch_price[idx], 1e-6),
            "switch_price mismatch at {idx}: cpu={} cuda={}",
            cpu.switch_price[idx],
            got_switch_price[idx]
        );
        assert!(
            approx_eq(cpu.trend[idx], got_trend[idx], 1e-9),
            "trend mismatch at {idx}: cpu={} cuda={}",
            cpu.trend[idx],
            got_trend[idx]
        );
        assert!(
            approx_eq(cpu.changed[idx], got_changed[idx], 1e-9),
            "changed mismatch at {idx}: cpu={} cuda={}",
            cpu.changed[idx],
            got_changed[idx]
        );
    }

    Ok(())
}
