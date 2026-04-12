use vector_ta::indicators::ehlers_data_sampling_relative_strength_indicator::{
    ehlers_data_sampling_relative_strength_indicator_batch_with_kernel,
    EhlersDataSamplingRelativeStrengthIndicatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaEhlersDataSamplingRelativeStrengthIndicator};

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
fn ehlers_data_sampling_relative_strength_indicator_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[ehlers_data_sampling_relative_strength_indicator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 1600usize;
    let mut open = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 102.0f64;
    for i in 7..len {
        let x = i as f64;
        base += (x * 0.013).sin() * 0.41 + (x * 0.005).cos() * 0.16;
        close[i] = base + (x * 0.021).sin() * 0.73;
        open[i] = close[i] - 0.22 - (x * 0.017).cos() * 0.18;
    }
    for i in (420..470).step_by(10) {
        open[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1080..1135).step_by(9) {
        open[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = EhlersDataSamplingRelativeStrengthIndicatorBatchRange {
        length: (10, 14, 2),
    };
    let cpu = ehlers_data_sampling_relative_strength_indicator_batch_with_kernel(
        &open,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaEhlersDataSamplingRelativeStrengthIndicator::new(0)
        .expect("CudaEhlersDataSamplingRelativeStrengthIndicator::new");
    let result = cuda.batch_dev(&open, &close, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_ds = vec![0.0f64; result.outputs.ds_rsi.len()];
    let mut got_original = vec![0.0f64; result.outputs.original_rsi.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.ds_rsi.buf.copy_to(&mut got_ds)?;
    result.outputs.original_rsi.buf.copy_to(&mut got_original)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.ds_rsi.len() {
        assert!(
            approx_eq(cpu.ds_rsi[idx], got_ds[idx], 1e-10),
            "ds_rsi mismatch at {idx}: cpu={} cuda={}",
            cpu.ds_rsi[idx],
            got_ds[idx]
        );
        assert!(
            approx_eq(cpu.original_rsi[idx], got_original[idx], 1e-10),
            "original_rsi mismatch at {idx}: cpu={} cuda={}",
            cpu.original_rsi[idx],
            got_original[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-10),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
