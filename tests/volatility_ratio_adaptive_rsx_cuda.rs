use vector_ta::indicators::volatility_ratio_adaptive_rsx::{
    volatility_ratio_adaptive_rsx_batch_with_kernel, VolatilityRatioAdaptiveRsxBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVolatilityRatioAdaptiveRsx};

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
fn volatility_ratio_adaptive_rsx_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[volatility_ratio_adaptive_rsx_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2080usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 77.0f64;
    for i in 14..len {
        let x = i as f64;
        base *= 1.0 + (x * 0.0012).sin() * 0.0021 + (x * 0.0008).cos() * 0.0011;
        data[i] = base + (x * 0.016).sin() * 0.74 + (x * 0.007).cos() * 0.21;
    }
    for i in (460..530).step_by(9) {
        data[i] = f64::NAN;
    }
    for i in (1320..1400).step_by(11) {
        data[i] = f64::NAN;
    }

    let sweep = VolatilityRatioAdaptiveRsxBatchRange {
        period: (10, 14, 2),
        speed: (0.3, 0.5, 0.2),
    };
    let cpu = volatility_ratio_adaptive_rsx_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaVolatilityRatioAdaptiveRsx::new(0).expect("CudaVolatilityRatioAdaptiveRsx::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_line = vec![0.0f64; result.outputs.line.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.line.buf.copy_to(&mut got_line)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.line.len() {
        assert!(
            approx_eq(cpu.line[idx], got_line[idx], 1e-9),
            "line mismatch at {idx}: cpu={} cuda={}",
            cpu.line[idx],
            got_line[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-9),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
