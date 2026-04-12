use vector_ta::indicators::monotonicity_index::{
    monotonicity_index_batch_with_kernel, MonotonicityIndexBatchRange, MonotonicityIndexMode,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaMonotonicityIndex};

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
fn monotonicity_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[monotonicity_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1536usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 50.0f64;
    for i in 5..len {
        value += (i as f64 * 0.012).sin() * 0.42 + (i as f64 * 0.009).cos() * 0.17;
        data[i] = value + (i as f64 * 0.004).sin() * 0.13;
    }

    let sweep = MonotonicityIndexBatchRange {
        length: (16, 20, 2),
        index_smooth: (3, 5, 2),
        mode: MonotonicityIndexMode::Efficiency,
    };
    let cpu = monotonicity_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaMonotonicityIndex::new(0).expect("CudaMonotonicityIndex::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_index = vec![0f64; result.outputs.index.len()];
    let mut got_mean = vec![0f64; result.outputs.cumulative_mean.len()];
    let mut got_upper = vec![0f64; result.outputs.upper_bound.len()];
    result.outputs.index.buf.copy_to(&mut got_index)?;
    result.outputs.cumulative_mean.buf.copy_to(&mut got_mean)?;
    result.outputs.upper_bound.buf.copy_to(&mut got_upper)?;

    for idx in 0..cpu.index.len() {
        assert!(
            approx_eq(cpu.index[idx], got_index[idx], 1e-9),
            "index mismatch at {idx}: cpu={} cuda={}",
            cpu.index[idx],
            got_index[idx]
        );
        assert!(
            approx_eq(cpu.cumulative_mean[idx], got_mean[idx], 1e-9),
            "mean mismatch at {idx}: cpu={} cuda={}",
            cpu.cumulative_mean[idx],
            got_mean[idx]
        );
        assert!(
            approx_eq(cpu.upper_bound[idx], got_upper[idx], 1e-9),
            "upper mismatch at {idx}: cpu={} cuda={}",
            cpu.upper_bound[idx],
            got_upper[idx]
        );
    }

    Ok(())
}
