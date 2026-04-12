use vector_ta::indicators::volatility_quality_index::{
    volatility_quality_index_batch_with_kernel, VolatilityQualityIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVolatilityQualityIndex};

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
fn volatility_quality_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[volatility_quality_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 3072usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 100.0f64;
    for i in 8..len {
        base += (i as f64 * 0.013).sin() * 0.72 + (i as f64 * 0.007).cos() * 0.19;
        let o = base + (i as f64 * 0.009).sin() * 0.21;
        let c = base + (i as f64 * 0.015).cos() * 0.24;
        let h = o.max(c) + 0.85 + (i as f64 * 0.006).sin().abs() * 0.23;
        let l = o.min(c) - 0.78 - (i as f64 * 0.012).cos().abs() * 0.27;
        open[i] = o;
        high[i] = h;
        low[i] = l;
        close[i] = c;
    }
    for i in (1200..1260).step_by(13) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    close[2100] = f64::NAN;

    let sweep = VolatilityQualityIndexBatchRange {
        fast_length: (5, 7, 2),
        slow_length: (19, 23, 4),
    };
    let cpu = volatility_quality_index_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaVolatilityQualityIndex::new(0).expect("CudaVolatilityQualityIndex::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_vqi = vec![0f64; result.outputs.vqi_sum.len()];
    let mut got_fast = vec![0f64; result.outputs.fast_sma.len()];
    let mut got_slow = vec![0f64; result.outputs.slow_sma.len()];
    result.outputs.vqi_sum.buf.copy_to(&mut got_vqi)?;
    result.outputs.fast_sma.buf.copy_to(&mut got_fast)?;
    result.outputs.slow_sma.buf.copy_to(&mut got_slow)?;

    for idx in 0..cpu.vqi_sum.len() {
        assert!(
            approx_eq(cpu.vqi_sum[idx], got_vqi[idx], 1e-10),
            "vqi_sum mismatch at {idx}: cpu={} cuda={}",
            cpu.vqi_sum[idx],
            got_vqi[idx]
        );
        assert!(
            approx_eq(cpu.fast_sma[idx], got_fast[idx], 1e-10),
            "fast_sma mismatch at {idx}: cpu={} cuda={}",
            cpu.fast_sma[idx],
            got_fast[idx]
        );
        assert!(
            approx_eq(cpu.slow_sma[idx], got_slow[idx], 1e-10),
            "slow_sma mismatch at {idx}: cpu={} cuda={}",
            cpu.slow_sma[idx],
            got_slow[idx]
        );
    }

    Ok(())
}
