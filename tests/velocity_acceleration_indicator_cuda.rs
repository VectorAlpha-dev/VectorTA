use vector_ta::indicators::velocity_acceleration_indicator::{
    velocity_acceleration_indicator_batch_with_kernel, VelocityAccelerationIndicatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVelocityAccelerationIndicator};

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
fn velocity_acceleration_indicator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[velocity_acceleration_indicator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 4096usize;
    let mut data = vec![f64::NAN; len];
    let mut value = 100.0f64;
    for i in 9..len {
        value += (i as f64 * 0.012).sin() * 0.57 + (i as f64 * 0.004).cos() * 0.22;
        data[i] = value + (i as f64 * 0.019).cos() * 0.31;
    }
    for i in (1400..1470).step_by(10) {
        data[i] = f64::NAN;
    }
    for i in (2800..2860).step_by(15) {
        data[i] = f64::NAN;
    }

    let sweep = VelocityAccelerationIndicatorBatchRange {
        length: (21, 25, 2),
        smooth_length: (4, 5, 1),
    };
    let cpu =
        velocity_acceleration_indicator_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda =
        CudaVelocityAccelerationIndicator::new(0).expect("CudaVelocityAccelerationIndicator::new");
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
