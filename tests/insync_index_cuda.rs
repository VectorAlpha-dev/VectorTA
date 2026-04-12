use vector_ta::indicators::insync_index::{insync_index_batch_with_kernel, InsyncIndexBatchRange};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::{CopyDestination, DeviceBuffer};
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaInsyncIndex};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlcv(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = vec![0.0; len];
    let mut low = vec![0.0; len];
    let mut close = vec![0.0; len];
    let mut volume = vec![0.0; len];
    let mut base = 101.0f64;
    for i in 0..len {
        let x = i as f64;
        base += (x * 0.007).sin() * 0.24 + (x * 0.0015).cos() * 0.09;
        let c = base + (x * 0.021).sin() * 0.93 + (x * 0.014).cos() * 0.31;
        let span = 0.86 + (x * 0.019).sin().abs() * 0.37;
        close[i] = c;
        high[i] = c + span;
        low[i] = c - span * (0.79 + (x * 0.012).cos().abs() * 0.21);
        volume[i] = 28_000.0 + (x * 0.017).sin() * 2_700.0 + (x % 19.0) * 107.0;
    }
    (high, low, close, volume)
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
fn insync_index_cuda_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[insync_index_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (high, low, close, volume) = sample_ohlcv(720);
    let sweep = InsyncIndexBatchRange {
        fast_length: (5, 7, 2),
        bb_multiplier: (1.8, 2.2, 0.4),
        ..InsyncIndexBatchRange::default()
    };
    let cpu =
        insync_index_batch_with_kernel(&high, &low, &close, &volume, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaInsyncIndex::new(0)?;
    let result = cuda.batch_dev(&high, &low, &close, &volume, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());
    assert_device_matches(&cpu.values, &result.outputs.values.buf)?;
    Ok(())
}
