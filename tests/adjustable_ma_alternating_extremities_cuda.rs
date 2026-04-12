use vector_ta::indicators::adjustable_ma_alternating_extremities::{
    adjustable_ma_alternating_extremities_batch_with_kernel,
    AdjustableMaAlternatingExtremitiesBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaAdjustableMaAlternatingExtremities};

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
fn adjustable_ma_alternating_extremities_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[adjustable_ma_alternating_extremities_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 540usize;
    let mut close = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut base = 82.0f64;
    for i in 30..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.28 + (x * 0.004).cos() * 0.09;
        close[i] = base + (x * 0.063).sin() * 1.4 + (x * 0.021).cos() * 0.6;
        let span = 1.0 + (x * 0.017).sin().abs() * 0.7;
        high[i] = close[i] + span;
        low[i] = close[i] - span * (0.8 + (x * 0.031).cos().abs() * 0.2);
    }

    let sweep = AdjustableMaAlternatingExtremitiesBatchRange {
        length: (14, 16, 2),
        mult: (1.5, 2.0, 0.5),
        alpha: (0.5, 1.0, 0.5),
        beta: (0.5, 1.0, 0.5),
    };

    let cpu = adjustable_ma_alternating_extremities_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaAdjustableMaAlternatingExtremities::new(0)
        .expect("CudaAdjustableMaAlternatingExtremities::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_ma = vec![0.0f64; result.outputs.ma.len()];
    let mut got_upper = vec![0.0f64; result.outputs.upper.len()];
    let mut got_lower = vec![0.0f64; result.outputs.lower.len()];
    let mut got_extremity = vec![0.0f64; result.outputs.extremity.len()];
    let mut got_state = vec![0.0f64; result.outputs.state.len()];
    let mut got_changed = vec![0.0f64; result.outputs.changed.len()];
    let mut got_smoothed_open = vec![0.0f64; result.outputs.smoothed_open.len()];
    let mut got_smoothed_high = vec![0.0f64; result.outputs.smoothed_high.len()];
    let mut got_smoothed_low = vec![0.0f64; result.outputs.smoothed_low.len()];
    let mut got_smoothed_close = vec![0.0f64; result.outputs.smoothed_close.len()];
    result.outputs.ma.buf.copy_to(&mut got_ma)?;
    result.outputs.upper.buf.copy_to(&mut got_upper)?;
    result.outputs.lower.buf.copy_to(&mut got_lower)?;
    result.outputs.extremity.buf.copy_to(&mut got_extremity)?;
    result.outputs.state.buf.copy_to(&mut got_state)?;
    result.outputs.changed.buf.copy_to(&mut got_changed)?;
    result
        .outputs
        .smoothed_open
        .buf
        .copy_to(&mut got_smoothed_open)?;
    result
        .outputs
        .smoothed_high
        .buf
        .copy_to(&mut got_smoothed_high)?;
    result
        .outputs
        .smoothed_low
        .buf
        .copy_to(&mut got_smoothed_low)?;
    result
        .outputs
        .smoothed_close
        .buf
        .copy_to(&mut got_smoothed_close)?;

    for idx in 0..cpu.ma.len() {
        assert!(
            approx_eq(cpu.ma[idx], got_ma[idx], 1e-6),
            "ma mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.upper[idx], got_upper[idx], 1e-6),
            "upper mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.lower[idx], got_lower[idx], 1e-6),
            "lower mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.extremity[idx], got_extremity[idx], 1e-6),
            "extremity mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.state[idx], got_state[idx], 1e-6),
            "state mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.changed[idx], got_changed[idx], 1e-6),
            "changed mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.smoothed_open[idx], got_smoothed_open[idx], 1e-6),
            "smoothed_open mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.smoothed_high[idx], got_smoothed_high[idx], 1e-6),
            "smoothed_high mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.smoothed_low[idx], got_smoothed_low[idx], 1e-6),
            "smoothed_low mismatch at {idx}"
        );
        assert!(
            approx_eq(cpu.smoothed_close[idx], got_smoothed_close[idx], 1e-6),
            "smoothed_close mismatch at {idx}"
        );
    }

    Ok(())
}
