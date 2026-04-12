use vector_ta::indicators::squeeze_index::{
    squeeze_index_batch_with_kernel, SqueezeIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaSqueezeIndex};

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
fn squeeze_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[squeeze_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2200usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 102.0f64;
    for i in 12..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.38 + (x * 0.003).cos() * 0.14;
        data[i] = base + (x * 0.021).sin() * 0.71 + (x * 0.005).cos() * 0.23;
    }
    for i in (480..560).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1360..1440).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = SqueezeIndexBatchRange {
        conv: (35.0, 55.0, 10.0),
        length: (10, 14, 2),
    };
    let cpu = squeeze_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaSqueezeIndex::new(0).expect("CudaSqueezeIndex::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-6),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
