use vector_ta::indicators::velocity_acceleration_convergence_divergence_indicator::{
    velocity_acceleration_convergence_divergence_indicator_batch_with_kernel,
    VelocityAccelerationConvergenceDivergenceIndicatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVelocityAccelerationConvergenceDivergenceIndicator};

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
fn velocity_acceleration_convergence_divergence_indicator_cuda_batch_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!(
            "[velocity_acceleration_convergence_divergence_indicator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let len = 2144usize;
    let mut data = vec![f64::NAN; len];
    let mut base = 97.0f64;
    for i in 9..len {
        let x = i as f64;
        base += (x * 0.012).sin() * 0.44 + (x * 0.004).cos() * 0.17;
        data[i] = base + (x * 0.019).sin() * 0.72 + (x * 0.006).cos() * 0.24;
    }
    for i in (470..540).step_by(8) {
        data[i] = f64::NAN;
    }
    for i in (1360..1435).step_by(10) {
        data[i] = f64::NAN;
    }

    let sweep = VelocityAccelerationConvergenceDivergenceIndicatorBatchRange {
        length: (15, 21, 6),
        smooth_length: (4, 6, 2),
    };
    let cpu = velocity_acceleration_convergence_divergence_indicator_batch_with_kernel(
        &data,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaVelocityAccelerationConvergenceDivergenceIndicator::new(0)
        .expect("CudaVelocityAccelerationConvergenceDivergenceIndicator::new");
    let result = cuda.batch_dev(&data, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_vacd = vec![0.0f64; result.outputs.vacd.len()];
    let mut got_signal = vec![0.0f64; result.outputs.signal.len()];
    result.outputs.vacd.buf.copy_to(&mut got_vacd)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.vacd.len() {
        assert!(
            approx_eq(cpu.vacd[idx], got_vacd[idx], 1e-6),
            "vacd mismatch at {idx}: cpu={} cuda={}",
            cpu.vacd[idx],
            got_vacd[idx]
        );
        assert!(
            approx_eq(cpu.signal[idx], got_signal[idx], 1e-6),
            "signal mismatch at {idx}: cpu={} cuda={}",
            cpu.signal[idx],
            got_signal[idx]
        );
    }

    Ok(())
}
