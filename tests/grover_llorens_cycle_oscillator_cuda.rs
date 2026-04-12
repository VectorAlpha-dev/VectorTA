use vector_ta::indicators::grover_llorens_cycle_oscillator::{
    grover_llorens_cycle_oscillator_batch_with_kernel, GroverLlorensCycleOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaGroverLlorensCycleOscillator};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut prev = 100.0f64;

    for i in 10..len {
        let x = i as f64;
        let wave = (x * 0.11).sin() * 2.1 + (x * 0.037).cos() * 1.4;
        let o = prev + wave * 0.32;
        let c = o + (x * 0.19).sin() * 0.95 - (x * 0.07).cos() * 0.38;
        let h = o.max(c) + 0.58 + (x * 0.03).sin().abs() * 0.24;
        let l = o.min(c) - 0.56 - (x * 0.02).cos().abs() * 0.22;
        open[i] = o;
        high[i] = h;
        low[i] = l;
        close[i] = c;
        prev = c;
    }

    for i in (280..350).step_by(13) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (860..930).step_by(17) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    (open, high, low, close)
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn grover_llorens_cycle_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>>
{
    if !cuda_available() {
        eprintln!(
            "[grover_llorens_cycle_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device"
        );
        return Ok(());
    }

    let (open, high, low, close) = sample_ohlc(1408);
    let sweep = GroverLlorensCycleOscillatorBatchRange {
        length: (32, 48, 16),
        mult: (6.0, 10.0, 4.0),
        source: "hlc3".to_string(),
        smooth: true,
        rsi_period: (12, 16, 4),
    };
    let cpu = grover_llorens_cycle_oscillator_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda =
        CudaGroverLlorensCycleOscillator::new(0).expect("CudaGroverLlorensCycleOscillator::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows, cpu.rows);
    assert_eq!(result.outputs.cols, cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_values = vec![0.0f64; result.outputs.len()];
    result.outputs.buf.copy_to(&mut got_values)?;

    for idx in 0..cpu.values.len() {
        assert!(
            approx_eq(cpu.values[idx], got_values[idx], 1e-6),
            "values mismatch at {idx}: cpu={} cuda={}",
            cpu.values[idx],
            got_values[idx]
        );
    }

    Ok(())
}
