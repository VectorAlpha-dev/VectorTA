use vector_ta::indicators::gopalakrishnan_range_index::{
    gopalakrishnan_range_index_batch_with_kernel, GopalakrishnanRangeIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaGopalakrishnanRangeIndex};

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
fn gopalakrishnan_range_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[gopalakrishnan_range_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        let center = 100.0 + 0.02 * x + (x * 0.013).sin();
        let spread = 1.0 + 0.3 * (x * 0.021).cos().abs();
        high[i] = center + spread;
        low[i] = center - spread;
    }
    for i in (1000..1080).step_by(13) {
        high[i] = f64::NAN;
    }
    for i in (1400..1480).step_by(11) {
        low[i] = f64::NAN;
    }

    let sweep = GopalakrishnanRangeIndexBatchRange { length: (3, 15, 4) };

    let cpu =
        gopalakrishnan_range_index_batch_with_kernel(&high, &low, &sweep, Kernel::ScalarBatch)?;
    let high_f32: Vec<f32> = high.iter().map(|&v| v as f32).collect();
    let low_f32: Vec<f32> = low.iter().map(|&v| v as f32).collect();
    let cuda = CudaGopalakrishnanRangeIndex::new(0).expect("CudaGopalakrishnanRangeIndex::new");
    let result = cuda
        .batch_dev(&high_f32, &low_f32, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0f32; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx] as f64, 1e-5),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
