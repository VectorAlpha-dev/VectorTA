use vector_ta::indicators::market_meanness_index::{
    market_meanness_index_batch_with_kernel, MarketMeannessIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMarketMeannessIndex};

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
fn market_meanness_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[market_meanness_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1536usize;
    let mut open = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 80.0f64;
    for i in 4..len {
        base += (i as f64 * 0.012).sin() * 0.54 + (i as f64 * 0.008).cos() * 0.22;
        let gap = (i as f64 * 0.006).cos() * 0.17;
        open[i] = base - 0.25 + gap;
        close[i] = base + 0.28 + (i as f64 * 0.015).sin() * 0.21;
    }

    let sweep = MarketMeannessIndexBatchRange {
        length: (18, 22, 2),
        source_mode: Some("Change".to_string()),
    };
    let cpu = market_meanness_index_batch_with_kernel(&open, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaMarketMeannessIndex::new(0).expect("CudaMarketMeannessIndex::new");
    let result = cuda.batch_dev(&open, &close, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_mmi = vec![0f64; result.outputs.mmi.len()];
    let mut got_smoothed = vec![0f64; result.outputs.mmi_smoothed.len()];
    result.outputs.mmi.buf.copy_to(&mut got_mmi)?;
    result.outputs.mmi_smoothed.buf.copy_to(&mut got_smoothed)?;

    for idx in 0..cpu.mmi.len() {
        assert!(
            approx_eq(cpu.mmi[idx], got_mmi[idx], 1e-9),
            "mmi mismatch at {idx}: cpu={} cuda={}",
            cpu.mmi[idx],
            got_mmi[idx]
        );
        assert!(
            approx_eq(cpu.mmi_smoothed[idx], got_smoothed[idx], 1e-9),
            "smoothed mismatch at {idx}: cpu={} cuda={}",
            cpu.mmi_smoothed[idx],
            got_smoothed[idx]
        );
    }

    Ok(())
}
