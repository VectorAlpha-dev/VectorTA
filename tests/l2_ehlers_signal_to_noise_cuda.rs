use vector_ta::indicators::l2_ehlers_signal_to_noise::{
    l2_ehlers_signal_to_noise_batch_with_kernel, L2EhlersSignalToNoiseBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaL2EhlersSignalToNoise};

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
fn l2_ehlers_signal_to_noise_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[l2_ehlers_signal_to_noise_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 2048usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut source = vec![f64::NAN; len];
    let mut base = 96.0f64;
    for i in 10..len {
        let x = i as f64;
        base += (x * 0.013).sin() * 0.41 + (x * 0.004).cos() * 0.17;
        let center = base + (x * 0.019).sin() * 0.58;
        high[i] = center + 0.92 + (x * 0.011).cos().abs() * 0.18;
        low[i] = center - 0.89 - (x * 0.009).sin().abs() * 0.16;
        source[i] = center + (x * 0.017).sin() * 0.12;
    }
    for i in (520..590).step_by(9) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        source[i] = f64::NAN;
    }
    for i in (1330..1410).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        source[i] = f64::NAN;
    }

    let sweep = L2EhlersSignalToNoiseBatchRange {
        smooth_period: (7, 11, 2),
    };
    let cpu = l2_ehlers_signal_to_noise_batch_with_kernel(
        &source,
        &high,
        &low,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaL2EhlersSignalToNoise::new(0).expect("CudaL2EhlersSignalToNoise::new");
    let result = cuda
        .batch_dev(&source, &high, &low, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got[idx], 1e-6),
            "mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got[idx]
        );
    }

    Ok(())
}
