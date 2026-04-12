use vector_ta::indicators::advance_decline_line::{
    advance_decline_line_batch_with_kernel, AdvanceDeclineLineBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAdvanceDeclineLine};

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
fn advance_decline_line_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[advance_decline_line_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    for i in 8..len {
        let x = i as f64;
        data[i] = (x * 0.021).sin() * 1.5 + (x * 0.013).cos() * 0.25;
    }
    for i in (600..700).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = AdvanceDeclineLineBatchRange;
    let cpu = advance_decline_line_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaAdvanceDeclineLine::new(0).expect("CudaAdvanceDeclineLine::new");
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
