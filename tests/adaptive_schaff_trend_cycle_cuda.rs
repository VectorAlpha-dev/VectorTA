use vector_ta::indicators::adaptive_schaff_trend_cycle::{
    adaptive_schaff_trend_cycle_batch_with_kernel, AdaptiveSchaffTrendCycleBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAdaptiveSchaffTrendCycle};

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
fn adaptive_schaff_trend_cycle_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[adaptive_schaff_trend_cycle_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 640usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 103.0f64;
    for i in 20..len {
        let x = i as f64;
        base += (x * 0.0075).sin() * 0.26 + (x * 0.0017).cos() * 0.06;
        close[i] = base + (x * 0.041).sin() * 1.45 + (x * 0.027).cos() * 0.75;
        let span = 1.0 + (x * 0.015).sin().abs() * 0.7;
        high[i] = close[i] + span;
        low[i] = close[i] - span * (0.82 + (x * 0.022).cos().abs() * 0.2);
    }
    high[358] = f64::NAN;
    low[358] = f64::NAN;
    close[358] = f64::NAN;

    let sweep = AdaptiveSchaffTrendCycleBatchRange {
        adaptive_length: (34, 38, 4),
        stc_length: (10, 12, 2),
        smoothing_factor: (0.35, 0.45, 0.10),
        fast_length: (23, 26, 3),
        slow_length: (45, 50, 5),
    };

    let cpu = adaptive_schaff_trend_cycle_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaAdaptiveSchaffTrendCycle::new(0).expect("CudaAdaptiveSchaffTrendCycle::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_stc = vec![0.0f64; result.outputs.stc.len()];
    let mut got_histogram = vec![0.0f64; result.outputs.histogram.len()];
    result.outputs.stc.buf.copy_to(&mut got_stc)?;
    result.outputs.histogram.buf.copy_to(&mut got_histogram)?;

    for idx in 0..cpu.stc.len() {
        assert!(
            approx_eq(cpu.stc[idx], got_stc[idx], 1e-6),
            "stc mismatch at {idx}: cpu={} cuda={}",
            cpu.stc[idx],
            got_stc[idx]
        );
        assert!(
            approx_eq(cpu.histogram[idx], got_histogram[idx], 1e-6),
            "histogram mismatch at {idx}: cpu={} cuda={}",
            cpu.histogram[idx],
            got_histogram[idx]
        );
    }

    Ok(())
}
