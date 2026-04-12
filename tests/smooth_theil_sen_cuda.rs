use vector_ta::indicators::smooth_theil_sen::{
    smooth_theil_sen_batch_with_kernel, SmoothTheilSenBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaSmoothTheilSen};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_data(len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    let mut base = 97.0f64;
    for (i, value) in out.iter_mut().enumerate() {
        let x = i as f64;
        base += (x * 0.006).sin() * 0.29 + (x * 0.0012).cos() * 0.07;
        *value = base + (x * 0.016).sin() * 0.77 + (x * 0.009).cos() * 0.23;
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
fn smooth_theil_sen_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[smooth_theil_sen_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let data = sample_data(512);
    let sweep = SmoothTheilSenBatchRange {
        length: (17, 19, 2),
        offset: (1, 2, 1),
        multiplier: (1.0, 1.5, 0.5),
        ..SmoothTheilSenBatchRange::default()
    };
    let cpu = smooth_theil_sen_batch_with_kernel(&data, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaSmoothTheilSen::new(0)?;
    let result = cuda.batch_dev(&data, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());
    assert_device_matches(&cpu.value, &result.outputs.value.buf)?;
    assert_device_matches(&cpu.upper, &result.outputs.upper.buf)?;
    assert_device_matches(&cpu.lower, &result.outputs.lower.buf)?;
    assert_device_matches(&cpu.slope, &result.outputs.slope.buf)?;
    assert_device_matches(&cpu.intercept, &result.outputs.intercept.buf)?;
    assert_device_matches(&cpu.deviation, &result.outputs.deviation.buf)?;
    Ok(())
}
