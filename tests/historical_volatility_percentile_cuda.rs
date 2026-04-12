use vector_ta::indicators::historical_volatility_percentile::{
    historical_volatility_percentile_batch_with_kernel, HistoricalVolatilityPercentileBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaHistoricalVolatilityPercentile};

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
fn historical_volatility_percentile_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[historical_volatility_percentile_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 10..len {
        value *= 1.0 + 0.0011 * (i as f64 * 0.009).sin() + 0.0002 * (i as f64 * 0.015).cos();
        data[i] = value.max(1.0);
    }
    for i in (600..680).step_by(19) {
        data[i] = f64::NAN;
    }
    data[1400] = f64::NAN;

    let sweep = HistoricalVolatilityPercentileBatchRange {
        length: (5, 7, 2),
        annual_length: (8, 10, 2),
    };

    let cpu =
        historical_volatility_percentile_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaHistoricalVolatilityPercentile::new(0)
        .expect("CudaHistoricalVolatilityPercentile::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_hvp = vec![0f64; result.outputs.hvp.len()];
    let mut got_hvp_sma = vec![0f64; result.outputs.hvp_sma.len()];
    result.outputs.hvp.buf.copy_to(&mut got_hvp)?;
    result.outputs.hvp_sma.buf.copy_to(&mut got_hvp_sma)?;

    for idx in 0..cpu.hvp.len() {
        assert!(
            approx_eq(cpu.hvp[idx], got_hvp[idx], 1e-10),
            "hvp mismatch at {idx}: cpu={} cuda={}",
            cpu.hvp[idx],
            got_hvp[idx]
        );
        assert!(
            approx_eq(cpu.hvp_sma[idx], got_hvp_sma[idx], 1e-10),
            "hvp_sma mismatch at {idx}: cpu={} cuda={}",
            cpu.hvp_sma[idx],
            got_hvp_sma[idx]
        );
    }

    Ok(())
}
