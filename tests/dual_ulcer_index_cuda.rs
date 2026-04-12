use vector_ta::indicators::dual_ulcer_index::{
    dual_ulcer_index_batch_with_kernel, DualUlcerIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDualUlcerIndex};

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
fn dual_ulcer_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[dual_ulcer_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 12..len {
        value *= 1.0 + 0.0010 * (i as f64 * 0.012).sin() + 0.0004 * (i as f64 * 0.019).cos();
        data[i] = value.max(1.0);
    }
    for i in (1000..1080).step_by(21) {
        data[i] = f64::NAN;
    }
    data[2200] = 0.0;

    let sweep = DualUlcerIndexBatchRange {
        period: (5, 9, 2),
        threshold: (0.1, 0.3, 0.2),
        auto_threshold: false,
    };

    let cpu = dual_ulcer_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaDualUlcerIndex::new(0).expect("CudaDualUlcerIndex::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_long = vec![0f64; result.outputs.long_ulcer.len()];
    let mut got_short = vec![0f64; result.outputs.short_ulcer.len()];
    let mut got_threshold = vec![0f64; result.outputs.threshold.len()];
    result.outputs.long_ulcer.buf.copy_to(&mut got_long)?;
    result.outputs.short_ulcer.buf.copy_to(&mut got_short)?;
    result.outputs.threshold.buf.copy_to(&mut got_threshold)?;

    for idx in 0..cpu.long_ulcer.len() {
        assert!(
            approx_eq(cpu.long_ulcer[idx], got_long[idx], 1e-10),
            "long_ulcer mismatch at {idx}: cpu={} cuda={}",
            cpu.long_ulcer[idx],
            got_long[idx]
        );
        assert!(
            approx_eq(cpu.short_ulcer[idx], got_short[idx], 1e-10),
            "short_ulcer mismatch at {idx}: cpu={} cuda={}",
            cpu.short_ulcer[idx],
            got_short[idx]
        );
        assert!(
            approx_eq(cpu.threshold[idx], got_threshold[idx], 1e-10),
            "threshold mismatch at {idx}: cpu={} cuda={}",
            cpu.threshold[idx],
            got_threshold[idx]
        );
    }

    Ok(())
}
