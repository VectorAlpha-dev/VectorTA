use vector_ta::indicators::ewma_volatility::{
    ewma_volatility_batch_with_kernel, EwmaVolatilityBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEwmaVolatility};

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
fn ewma_volatility_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[ewma_volatility_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 12..len {
        value *= 1.0 + 0.0014 * (i as f64 * 0.009).sin() + 0.0004 * (i as f64 * 0.015).cos();
        data[i] = value.max(1.0);
    }
    for i in (700..860).step_by(19) {
        data[i] = f64::NAN;
    }
    data[1900] = 0.0;

    let sweep = EwmaVolatilityBatchRange {
        lambda: (0.80, 0.94, 0.07),
    };

    let cpu = ewma_volatility_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaEwmaVolatility::new(0).expect("CudaEwmaVolatility::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-5),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
