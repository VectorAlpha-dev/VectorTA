use vector_ta::indicators::statistical_trailing_stop::{
    statistical_trailing_stop_batch_with_kernel, StatisticalTrailingStopBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaStatisticalTrailingStop};

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
fn statistical_trailing_stop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[statistical_trailing_stop_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2400usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 102.0f64;
    for i in 16..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.33 + (x * 0.003).cos() * 0.16;
        close[i] = base + (x * 0.018).sin() * 0.56 + (x * 0.007).cos() * 0.20;
        high[i] = close[i] + 0.86 + (x * 0.014).sin().abs() * 0.23;
        low[i] = close[i] - 0.84 - (x * 0.012).cos().abs() * 0.21;
    }
    for i in (500..590).step_by(13) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1620..1710).step_by(12) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = StatisticalTrailingStopBatchRange {
        data_length: (9, 10, 1),
        normalization_length: (10, 11, 1),
        base_level: ("level1".to_string(), "level2".to_string(), 1),
    };
    let cpu = statistical_trailing_stop_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaStatisticalTrailingStop::new(0).expect("CudaStatisticalTrailingStop::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_level = vec![0.0f64; result.outputs.level.len()];
    let mut got_anchor = vec![0.0f64; result.outputs.anchor.len()];
    let mut got_bias = vec![0.0f64; result.outputs.bias.len()];
    let mut got_changed = vec![0.0f64; result.outputs.changed.len()];
    result.outputs.level.buf.copy_to(&mut got_level)?;
    result.outputs.anchor.buf.copy_to(&mut got_anchor)?;
    result.outputs.bias.buf.copy_to(&mut got_bias)?;
    result.outputs.changed.buf.copy_to(&mut got_changed)?;

    for idx in 0..cpu.level.len() {
        assert!(
            approx_eq(cpu.level[idx], got_level[idx], 1e-6),
            "level mismatch at {idx}: cpu={} cuda={}",
            cpu.level[idx],
            got_level[idx]
        );
        assert!(
            approx_eq(cpu.anchor[idx], got_anchor[idx], 1e-6),
            "anchor mismatch at {idx}: cpu={} cuda={}",
            cpu.anchor[idx],
            got_anchor[idx]
        );
        assert!(
            approx_eq(cpu.bias[idx], got_bias[idx], 1e-9),
            "bias mismatch at {idx}: cpu={} cuda={}",
            cpu.bias[idx],
            got_bias[idx]
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
