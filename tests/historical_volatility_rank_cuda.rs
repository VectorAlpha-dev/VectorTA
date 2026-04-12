use vector_ta::indicators::historical_volatility_rank::{
    historical_volatility_rank_batch_with_kernel, HistoricalVolatilityRankBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaHistoricalVolatilityRank};

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
fn historical_volatility_rank_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[historical_volatility_rank_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 12..len {
        value *= 1.0 + 0.0012 * (i as f64 * 0.011).sin() + 0.0003 * (i as f64 * 0.017).cos();
        data[i] = value.max(1.0);
    }
    for i in (900..980).step_by(17) {
        data[i] = f64::NAN;
    }
    data[1800] = 0.0;

    let sweep = HistoricalVolatilityRankBatchRange {
        hv_length: (5, 10, 5),
        rank_length: (8, 12, 4),
        annualization_days: (252.0, 365.0, 113.0),
        bar_days: (1.0, 1.0, 0.0),
    };

    let cpu = historical_volatility_rank_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaHistoricalVolatilityRank::new(0).expect("CudaHistoricalVolatilityRank::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_hvr = vec![0f64; result.outputs.hvr.len()];
    let mut got_hv = vec![0f64; result.outputs.hv.len()];
    result.outputs.hvr.buf.copy_to(&mut got_hvr)?;
    result.outputs.hv.buf.copy_to(&mut got_hv)?;

    for idx in 0..cpu.hvr.len() {
        assert!(
            approx_eq(cpu.hvr[idx], got_hvr[idx], 1e-4),
            "hvr mismatch at {idx}: cpu={} cuda={}",
            cpu.hvr[idx],
            got_hvr[idx]
        );
        assert!(
            approx_eq(cpu.hv[idx], got_hv[idx], 1e-4),
            "hv mismatch at {idx}: cpu={} cuda={}",
            cpu.hv[idx],
            got_hv[idx]
        );
    }

    Ok(())
}
