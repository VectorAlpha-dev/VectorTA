use vector_ta::indicators::normalized_volume_true_range::{
    normalized_volume_true_range_batch_with_kernel, NormalizedVolumeTrueRangeBatchRange,
    NormalizedVolumeTrueRangeStyle,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaNormalizedVolumeTrueRange};

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
fn normalized_volume_true_range_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[normalized_volume_true_range_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1600usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut prev_close = 100.0f64;
    for i in 10..len {
        let x = i as f64;
        let drift = x * 0.018;
        let body = (x * 0.13).sin() * 1.45;
        let span = 2.2 + (x * 0.07).cos().abs();
        let open_value = prev_close + (x * 0.03).sin() * 0.55;
        let close_value = 100.0 + drift + body;
        let high_value = open_value.max(close_value) + span * 0.57;
        let low_value = open_value.min(close_value) - span * 0.43;
        let volume_value = 1_200_000.0 + (x * 0.17).sin().abs() * 310_000.0 + x * 950.0;
        open[i] = open_value;
        high[i] = high_value;
        low[i] = low_value;
        close[i] = close_value;
        volume[i] = volume_value;
        prev_close = close_value;
    }
    for i in (330..390).step_by(11) {
        volume[i] = f64::NAN;
    }
    for i in (980..1040).step_by(9) {
        open[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = NormalizedVolumeTrueRangeBatchRange {
        true_range_style: Some(NormalizedVolumeTrueRangeStyle::Body),
        outlier_range: (4.0, 5.0, 1.0),
        atr_length: (8, 10, 2),
        volume_length: (5, 7, 2),
    };
    let cpu = normalized_volume_true_range_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaNormalizedVolumeTrueRange::new(0).expect("CudaNormalizedVolumeTrueRange::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &volume, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_nv = vec![0.0f64; result.outputs.normalized_volume.len()];
    let mut got_ntr = vec![0.0f64; result.outputs.normalized_true_range.len()];
    let mut got_baseline = vec![0.0f64; result.outputs.baseline.len()];
    let mut got_atr = vec![0.0f64; result.outputs.atr.len()];
    let mut got_avg_volume = vec![0.0f64; result.outputs.average_volume.len()];
    result.outputs.normalized_volume.buf.copy_to(&mut got_nv)?;
    result
        .outputs
        .normalized_true_range
        .buf
        .copy_to(&mut got_ntr)?;
    result.outputs.baseline.buf.copy_to(&mut got_baseline)?;
    result.outputs.atr.buf.copy_to(&mut got_atr)?;
    result
        .outputs
        .average_volume
        .buf
        .copy_to(&mut got_avg_volume)?;

    for idx in 0..cpu.normalized_volume.len() {
        assert!(
            approx_eq(cpu.normalized_volume[idx], got_nv[idx], 1e-6),
            "normalized_volume mismatch at {idx}: cpu={} cuda={}",
            cpu.normalized_volume[idx],
            got_nv[idx]
        );
        assert!(
            approx_eq(cpu.normalized_true_range[idx], got_ntr[idx], 1e-6),
            "normalized_true_range mismatch at {idx}: cpu={} cuda={}",
            cpu.normalized_true_range[idx],
            got_ntr[idx]
        );
        assert!(
            approx_eq(cpu.baseline[idx], got_baseline[idx], 1e-6),
            "baseline mismatch at {idx}: cpu={} cuda={}",
            cpu.baseline[idx],
            got_baseline[idx]
        );
        assert!(
            approx_eq(cpu.atr[idx], got_atr[idx], 1e-6),
            "atr mismatch at {idx}: cpu={} cuda={}",
            cpu.atr[idx],
            got_atr[idx]
        );
        assert!(
            approx_eq(cpu.average_volume[idx], got_avg_volume[idx], 1e-6),
            "average_volume mismatch at {idx}: cpu={} cuda={}",
            cpu.average_volume[idx],
            got_avg_volume[idx]
        );
    }

    Ok(())
}
