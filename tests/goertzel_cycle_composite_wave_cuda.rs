use vector_ta::indicators::goertzel_cycle_composite_wave::{
    goertzel_cycle_composite_wave_batch_with_kernel, GoertzelCycleCompositeWaveBatchRange,
    GoertzelCycleCompositeWaveParams, GoertzelDetrendMode,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaGoertzelCycleCompositeWave};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_data(len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    let mut base = 100.0f64;
    for (i, value) in out.iter_mut().enumerate() {
        let x = i as f64;
        base += (x * 0.008).sin() * 0.31 + (x * 0.0017).cos() * 0.14;
        *value = base + (x * 0.029).sin() * 1.27 + (x * 0.011).cos() * 0.42;
    }
    out
}

#[cfg(feature = "cuda")]
fn assert_device_matches(
    expected: &[f64],
    buf: &DeviceBuffer<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut got = vec![0.0; expected.len()];
    buf.copy_to(&mut got)?;
    for idx in 0..expected.len() {
        assert!(approx_eq(expected[idx], got[idx], 1e-9));
    }
    Ok(())
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn goertzel_cycle_composite_wave_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[goertzel_cycle_composite_wave_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let data = sample_data(960);
    let sweep = GoertzelCycleCompositeWaveBatchRange {
        max_period: (48, 60, 12),
        start_at_cycle: (1, 2, 1),
        use_top_cycles: (1, 2, 1),
        base_params: GoertzelCycleCompositeWaveParams {
            detrend_mode: Some(GoertzelDetrendMode::ZeroLagDetrending),
            filter_bartels: Some(false),
            use_cosine: Some(false),
            ..GoertzelCycleCompositeWaveParams::default()
        },
    };

    let cpu = goertzel_cycle_composite_wave_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaGoertzelCycleCompositeWave::new(0)?;
    let result = cuda.batch_dev(&data, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());
    assert_device_matches(&cpu.values, &result.outputs.values.buf)?;
    Ok(())
}
