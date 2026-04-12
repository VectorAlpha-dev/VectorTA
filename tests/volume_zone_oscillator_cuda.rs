use vector_ta::indicators::volume_zone_oscillator::{
    volume_zone_oscillator_batch_with_kernel, VolumeZoneOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaVolumeZoneOscillator};

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
fn volume_zone_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[volume_zone_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut price = 100.0f64;
    for i in 7..len {
        price += (i as f64 * 0.017).sin() * 0.44 + (i as f64 * 0.009).cos() * 0.18;
        close[i] = price + (i as f64 * 0.005).sin() * 0.07;
        volume[i] = 6000.0 + (i as f64 * 0.023).sin() * 850.0 + (i % 21) as f64 * 27.0;
    }
    for i in (900..960).step_by(11) {
        close[i] = f64::NAN;
    }
    for i in (1700..1775).step_by(13) {
        volume[i] = f64::NAN;
    }
    for i in (2500..2555).step_by(17) {
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let sweep = VolumeZoneOscillatorBatchRange {
        length: (10, 14, 2),
        noise_filter: (3, 5, 2),
        intraday_smoothing: Some(true),
    };
    let cpu =
        volume_zone_oscillator_batch_with_kernel(&close, &volume, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaVolumeZoneOscillator::new(0).expect("CudaVolumeZoneOscillator::new");
    let result = cuda.batch_dev(&close, &volume, &sweep).expect("batch_dev");

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
