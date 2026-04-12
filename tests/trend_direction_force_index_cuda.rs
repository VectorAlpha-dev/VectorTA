use vector_ta::indicators::trend_direction_force_index::{
    trend_direction_force_index_batch_with_kernel, TrendDirectionForceIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaTrendDirectionForceIndex};

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
fn trend_direction_force_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trend_direction_force_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 6..len {
        value += (i as f64 * 0.013).sin() * 0.63 + (i as f64 * 0.007).cos() * 0.28;
        data[i] = value;
    }
    for i in (1100..1160).step_by(17) {
        data[i] = f64::NAN;
    }
    for i in (2500..2580).step_by(19) {
        data[i] = f64::NAN;
    }

    let sweep = TrendDirectionForceIndexBatchRange { length: (8, 14, 3) };
    let cpu = trend_direction_force_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaTrendDirectionForceIndex::new(0).expect("CudaTrendDirectionForceIndex::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0f64; result.outputs.len()];
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
