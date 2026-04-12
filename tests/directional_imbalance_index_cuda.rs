use vector_ta::indicators::directional_imbalance_index::{
    directional_imbalance_index_batch_with_kernel, DirectionalImbalanceIndexBatchRange,
    DirectionalImbalanceIndexParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaDirectionalImbalanceIndex};

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
fn directional_imbalance_index_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[directional_imbalance_index_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1600usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut base = 100.0f64;
    for i in 5..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.58 + (x * 0.007).cos() * 0.23;
        let center = base + (x * 0.021).sin() * 0.17;
        high[i] = center + 0.9 + (x * 0.011).cos().abs() * 0.21;
        low[i] = center - 0.8 - (x * 0.009).sin().abs() * 0.19;
    }
    for i in (520..590).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
    }
    for i in (1180..1260).step_by(11) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
    }

    let sweep = DirectionalImbalanceIndexBatchRange {
        length: (8, 12, 2),
        period: (30, 40, 10),
    };
    let fixed = DirectionalImbalanceIndexParams {
        length: None,
        period: None,
    };
    let cpu = directional_imbalance_index_batch_with_kernel(
        &high,
        &low,
        &sweep,
        &fixed,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaDirectionalImbalanceIndex::new(0).expect("CudaDirectionalImbalanceIndex::new");
    let result = cuda
        .batch_dev(&high, &low, &sweep, &fixed)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_up = vec![0.0f64; result.outputs.up.len()];
    let mut got_down = vec![0.0f64; result.outputs.down.len()];
    let mut got_bulls = vec![0.0f64; result.outputs.bulls.len()];
    let mut got_bears = vec![0.0f64; result.outputs.bears.len()];
    let mut got_upper = vec![0.0f64; result.outputs.upper.len()];
    let mut got_lower = vec![0.0f64; result.outputs.lower.len()];
    result.outputs.up.buf.copy_to(&mut got_up)?;
    result.outputs.down.buf.copy_to(&mut got_down)?;
    result.outputs.bulls.buf.copy_to(&mut got_bulls)?;
    result.outputs.bears.buf.copy_to(&mut got_bears)?;
    result.outputs.upper.buf.copy_to(&mut got_upper)?;
    result.outputs.lower.buf.copy_to(&mut got_lower)?;

    for idx in 0..cpu.up.len() {
        assert!(
            approx_eq(cpu.up[idx], got_up[idx], 1e-12),
            "up mismatch at {idx}: cpu={} cuda={}",
            cpu.up[idx],
            got_up[idx]
        );
        assert!(
            approx_eq(cpu.down[idx], got_down[idx], 1e-12),
            "down mismatch at {idx}: cpu={} cuda={}",
            cpu.down[idx],
            got_down[idx]
        );
        assert!(
            approx_eq(cpu.bulls[idx], got_bulls[idx], 1e-12),
            "bulls mismatch at {idx}: cpu={} cuda={}",
            cpu.bulls[idx],
            got_bulls[idx]
        );
        assert!(
            approx_eq(cpu.bears[idx], got_bears[idx], 1e-12),
            "bears mismatch at {idx}: cpu={} cuda={}",
            cpu.bears[idx],
            got_bears[idx]
        );
        assert!(
            approx_eq(cpu.upper[idx], got_upper[idx], 1e-12),
            "upper mismatch at {idx}: cpu={} cuda={}",
            cpu.upper[idx],
            got_upper[idx]
        );
        assert!(
            approx_eq(cpu.lower[idx], got_lower[idx], 1e-12),
            "lower mismatch at {idx}: cpu={} cuda={}",
            cpu.lower[idx],
            got_lower[idx]
        );
    }

    Ok(())
}
