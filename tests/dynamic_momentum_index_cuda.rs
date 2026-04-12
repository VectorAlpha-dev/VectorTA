use vector_ta::indicators::dynamic_momentum_index::{
    dynamic_momentum_index_batch_with_kernel, DynamicMomentumIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDynamicMomentumIndex};

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
fn dynamic_momentum_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dynamic_momentum_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 6..len {
        let x = i as f64;
        value += (x * 0.014).sin() * 0.57 + (x * 0.005).cos() * 0.21;
        data[i] = value + (x * 0.031).sin() * 0.16;
    }
    for i in (620..690).step_by(10) {
        data[i] = f64::NAN;
    }
    for i in (1500..1580).step_by(9) {
        data[i] = f64::NAN;
    }

    let sweep = DynamicMomentumIndexBatchRange {
        rsi_period: (12, 14, 2),
        volatility_period: (5, 7, 2),
        volatility_sma_period: (9, 9, 0),
        upper_limit: (24, 24, 0),
        lower_limit: (5, 5, 0),
    };
    let cpu = dynamic_momentum_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaDynamicMomentumIndex::new(0).expect("CudaDynamicMomentumIndex::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-10),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
