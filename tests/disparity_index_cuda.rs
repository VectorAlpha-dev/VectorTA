use vector_ta::indicators::disparity_index::{
    disparity_index_batch_with_kernel, DisparityIndexBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDisparityIndex};

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
fn disparity_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[disparity_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1792usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 110.0f64;
    for i in 8..len {
        let x = i as f64;
        value += (x * 0.017).sin() * 0.64 + (x * 0.005).cos() * 0.27;
        data[i] = value + (x * 0.029).sin() * 0.19;
    }
    for i in (460..520).step_by(10) {
        data[i] = f64::NAN;
    }
    for i in (1190..1260).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = DisparityIndexBatchRange {
        ema_period: (6, 8, 2),
        lookback_period: (5, 5, 0),
        smoothing_period: (3, 3, 0),
        smoothing_types: vec!["ema".to_string(), "sma".to_string()],
    };
    let cpu = disparity_index_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaDisparityIndex::new(0).expect("CudaDisparityIndex::new");
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
