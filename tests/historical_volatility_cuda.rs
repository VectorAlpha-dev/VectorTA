use vector_ta::indicators::historical_volatility::{
    historical_volatility_batch_with_kernel, HistoricalVolatilityBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaHistoricalVolatility};

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
fn historical_volatility_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[historical_volatility_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 6..len {
        value *= 1.0 + 0.001 * (i as f64 * 0.013).sin() + 0.0002 * (i as f64 * 0.007).cos();
        data[i] = value;
    }
    for i in (700..760).step_by(11) {
        data[i] = f64::NAN;
    }
    data[1500] = 0.0;

    let sweep = HistoricalVolatilityBatchRange {
        lookback: (5, 20, 5),
        annualization_days: (250.0, 252.0, 2.0),
    };

    let cpu = historical_volatility_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let cuda = CudaHistoricalVolatility::new(0).expect("CudaHistoricalVolatility::new");
    let result = cuda.batch_dev(&data_f32, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0f32; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx] as f64, 5e-3),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
