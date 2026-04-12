use vector_ta::indicators::random_walk_index::{
    random_walk_index_batch_with_kernel, RandomWalkIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRandomWalkIndex};

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
fn random_walk_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[random_walk_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 120.0f64;
    for i in 5..len {
        base += (i as f64 * 0.014).sin() * 0.61 + (i as f64 * 0.009).cos() * 0.24;
        let center = base + (i as f64 * 0.004).sin() * 0.18;
        high[i] = center + 1.3 + (i as f64 * 0.012).cos().abs() * 0.17;
        low[i] = center - 1.2 - (i as f64 * 0.007).sin().abs() * 0.19;
        close[i] = center + (i as f64 * 0.011).sin() * 0.26;
    }

    let sweep = RandomWalkIndexBatchRange {
        length: (10, 16, 3),
    };
    let cpu =
        random_walk_index_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaRandomWalkIndex::new(0).expect("CudaRandomWalkIndex::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_high = vec![0f64; result.outputs.high.len()];
    let mut got_low = vec![0f64; result.outputs.low.len()];
    result.outputs.high.buf.copy_to(&mut got_high)?;
    result.outputs.low.buf.copy_to(&mut got_low)?;

    for idx in 0..cpu.high.len() {
        assert!(
            approx_eq(cpu.high[idx], got_high[idx], 1e-10),
            "high mismatch at {idx}: cpu={} cuda={}",
            cpu.high[idx],
            got_high[idx]
        );
        assert!(
            approx_eq(cpu.low[idx], got_low[idx], 1e-10),
            "low mismatch at {idx}: cpu={} cuda={}",
            cpu.low[idx],
            got_low[idx]
        );
    }

    Ok(())
}
