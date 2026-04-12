use vector_ta::indicators::on_balance_volume_oscillator::{
    on_balance_volume_oscillator_batch_with_kernel, OnBalanceVolumeOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaOnBalanceVolumeOscillator};

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
fn on_balance_volume_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[on_balance_volume_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut source = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut price = 100.0f64;
    for i in 8..len {
        price += (i as f64 * 0.014).sin() * 0.70 + (i as f64 * 0.005).cos() * 0.22;
        source[i] = price;
        volume[i] = 5000.0 + (i as f64 * 0.031).sin() * 900.0 + (i % 17) as f64 * 23.0;
    }

    for i in (900..980).step_by(11) {
        source[i] = f64::NAN;
        volume[i] = f64::NAN;
    }
    for i in (2500..2545).step_by(9) {
        volume[i] = 0.0;
    }

    let sweep = OnBalanceVolumeOscillatorBatchRange {
        obv_length: (5, 15, 5),
        ema_length: (3, 5, 2),
    };
    let cpu = on_balance_volume_oscillator_batch_with_kernel(
        &source,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaOnBalanceVolumeOscillator::new(0).expect("CudaOnBalanceVolumeOscillator::new");
    let result = cuda.batch_dev(&source, &volume, &sweep).expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_line = vec![0f64; result.outputs.line.len()];
    let mut got_signal = vec![0f64; result.outputs.signal.len()];
    result.outputs.line.buf.copy_to(&mut got_line)?;
    result.outputs.signal.buf.copy_to(&mut got_signal)?;

    for idx in 0..cpu.line.len() {
        assert!(
            approx_eq(cpu.line[idx], got_line[idx], 1e-10),
            "line mismatch at {idx}: cpu={} cuda={}",
            cpu.line[idx],
            got_line[idx]
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
