use vector_ta::indicators::pretty_good_oscillator::{
    pretty_good_oscillator_batch_with_kernel, PrettyGoodOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaPrettyGoodOscillator};

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
fn pretty_good_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[pretty_good_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 4096usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut source = vec![f64::NAN; len];

    let mut base = 100.0f64;
    for i in 10..len {
        base += (i as f64 * 0.017).sin() * 0.45 + (i as f64 * 0.009).cos() * 0.18;
        let c = base + (i as f64 * 0.013).cos() * 0.35;
        let h = c + 1.10 + (i as f64 * 0.007).sin().abs() * 0.25;
        let l = c - 0.95 - (i as f64 * 0.011).cos().abs() * 0.20;
        high[i] = h;
        low[i] = l;
        close[i] = c;
        source[i] = (h + l + c) / 3.0;
    }

    for i in (700..780).step_by(9) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        source[i] = f64::NAN;
    }
    for i in (2000..2060).step_by(13) {
        source[i] = f64::NAN;
    }

    let sweep = PrettyGoodOscillatorBatchRange {
        length: (10, 30, 10),
    };
    let cpu = pretty_good_oscillator_batch_with_kernel(
        &high,
        &low,
        &close,
        &source,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaPrettyGoodOscillator::new(0).expect("CudaPrettyGoodOscillator::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &source, &sweep)
        .expect("batch_dev");

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
