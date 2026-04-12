use vector_ta::indicators::market_structure_trailing_stop::{
    market_structure_trailing_stop_batch_with_kernel, MarketStructureTrailingStopBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMarketStructureTrailingStop};

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
fn market_structure_trailing_stop_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[market_structure_trailing_stop_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2400usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 20..len {
        let x = i as f64;
        base += (x * 0.011).sin() * 0.46 + (x * 0.003).cos() * 0.21;
        close[i] = base + (x * 0.027).sin() * 1.35;
        open[i] = close[i] + (x * 0.019).cos() * 0.42;
        high[i] = open[i].max(close[i]) + 0.84 + (x * 0.012).sin().abs() * 0.31;
        low[i] = open[i].min(close[i]) - 0.79 - (x * 0.015).cos().abs() * 0.28;
    }
    for i in (680..760).step_by(13) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1610..1690).step_by(11) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = MarketStructureTrailingStopBatchRange {
        length: (5, 7, 2),
        increment_factor: (80.0, 120.0, 40.0),
    };
    let cpu = market_structure_trailing_stop_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        "All",
        Kernel::ScalarBatch,
    )?;
    let cuda =
        CudaMarketStructureTrailingStop::new(0).expect("CudaMarketStructureTrailingStop::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &sweep, "All")
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_trailing_stop = vec![0.0f64; result.outputs.trailing_stop.len()];
    let mut got_state = vec![0.0f64; result.outputs.state.len()];
    let mut got_structure = vec![0.0f64; result.outputs.structure.len()];
    result
        .outputs
        .trailing_stop
        .buf
        .copy_to(&mut got_trailing_stop)?;
    result.outputs.state.buf.copy_to(&mut got_state)?;
    result.outputs.structure.buf.copy_to(&mut got_structure)?;

    for idx in 0..cpu.trailing_stop.len() {
        assert!(
            approx_eq(cpu.trailing_stop[idx], got_trailing_stop[idx], 1e-9),
            "trailing_stop mismatch at {idx}: cpu={} cuda={}",
            cpu.trailing_stop[idx],
            got_trailing_stop[idx]
        );
        assert!(
            approx_eq(cpu.state[idx], got_state[idx], 1e-9),
            "state mismatch at {idx}: cpu={} cuda={}",
            cpu.state[idx],
            got_state[idx]
        );
        assert!(
            approx_eq(cpu.structure[idx], got_structure[idx], 1e-9),
            "structure mismatch at {idx}: cpu={} cuda={}",
            cpu.structure[idx],
            got_structure[idx]
        );
    }

    Ok(())
}
