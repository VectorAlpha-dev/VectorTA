use vector_ta::indicators::accumulation_swing_index::{
    accumulation_swing_index_batch_with_kernel, AccumulationSwingIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAccumulationSwingIndex};

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
fn accumulation_swing_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[accumulation_swing_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 100.0f64;
    for i in 8..len {
        let x = i as f64;
        base += (x * 0.013).sin() * 0.41 + (x * 0.004).cos() * 0.19;
        open[i] = base + (x * 0.007).cos() * 0.33;
        close[i] = base + (x * 0.017).sin() * 0.52;
        high[i] = open[i].max(close[i]) + 0.74 + (x * 0.012).sin().abs() * 0.21;
        low[i] = open[i].min(close[i]) - 0.71 - (x * 0.011).cos().abs() * 0.18;
    }
    for i in (420..520).step_by(11) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1260..1350).step_by(13) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = AccumulationSwingIndexBatchRange {
        daily_limit: (8_000.0, 12_000.0, 2_000.0),
    };
    let cpu = accumulation_swing_index_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaAccumulationSwingIndex::new(0).expect("CudaAccumulationSwingIndex::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-9),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
