use vector_ta::indicators::bull_power_vs_bear_power::{
    bull_power_vs_bear_power_batch_with_kernel, BullPowerVsBearPowerBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaBullPowerVsBearPower};

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
fn bull_power_vs_bear_power_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bull_power_vs_bear_power_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 100.0f64;
    for i in 7..len {
        base += (i as f64 * 0.016).sin() * 0.55 + (i as f64 * 0.005).cos() * 0.21;
        let o = base + (i as f64 * 0.009).sin() * 0.18;
        let c = base + (i as f64 * 0.013).cos() * 0.22;
        let h = o.max(c) + 0.8 + (i as f64 * 0.007).sin().abs() * 0.2;
        let l = o.min(c) - 0.7 - (i as f64 * 0.011).cos().abs() * 0.2;
        open[i] = o;
        high[i] = h;
        low[i] = l;
        close[i] = c;
    }
    for i in (900..980).step_by(11) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    close[1700] = 0.0;

    let sweep = BullPowerVsBearPowerBatchRange { period: (3, 7, 2) };
    let cpu = bull_power_vs_bear_power_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaBullPowerVsBearPower::new(0).expect("CudaBullPowerVsBearPower::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-10),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
