use vector_ta::indicators::standardized_psar_oscillator::{
    standardized_psar_oscillator_batch_with_kernel, StandardizedPsarOscillatorBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaStandardizedPsarOscillator};

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
fn standardized_psar_oscillator_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[standardized_psar_oscillator_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1792usize;
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.010).sin() * 0.31 + (x * 0.004).cos() * 0.19;
        let c = base + (x * 0.023).sin() * 5.8 + (x * 0.008).cos() * 2.2;
        close[i] = c;
        high[i] = c + 1.15 + (i % 3) as f64 * 0.06;
        low[i] = c - 1.04 - (i % 2) as f64 * 0.07;
    }

    for i in (420..470).step_by(12) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (1180..1236).step_by(10) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    let sweep = StandardizedPsarOscillatorBatchRange {
        start: (0.02, 0.03, 0.01),
        increment: (0.0005, 0.0005, 0.0),
        maximum: (0.2, 0.2, 0.0),
        standardization_length: (10, 11, 1),
        wma_length: (8, 10, 2),
        wma_lag: (2, 2, 0),
        pivot_left: (6, 6, 0),
        pivot_right: (1, 1, 0),
        plot_bullish: true,
        plot_bearish: true,
    };
    let cpu = standardized_psar_oscillator_batch_with_kernel(
        &high,
        &low,
        &close,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaStandardizedPsarOscillator::new(0).expect("CudaStandardizedPsarOscillator::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_oscillator = vec![0.0f64; result.outputs.oscillator.len()];
    let mut got_ma = vec![0.0f64; result.outputs.ma.len()];
    let mut got_bullish_reversal = vec![0.0f64; result.outputs.bullish_reversal.len()];
    let mut got_bearish_reversal = vec![0.0f64; result.outputs.bearish_reversal.len()];
    let mut got_regular_bullish = vec![0.0f64; result.outputs.regular_bullish.len()];
    let mut got_regular_bearish = vec![0.0f64; result.outputs.regular_bearish.len()];
    let mut got_bullish_weakening = vec![0.0f64; result.outputs.bullish_weakening.len()];
    let mut got_bearish_weakening = vec![0.0f64; result.outputs.bearish_weakening.len()];
    result.outputs.oscillator.buf.copy_to(&mut got_oscillator)?;
    result.outputs.ma.buf.copy_to(&mut got_ma)?;
    result
        .outputs
        .bullish_reversal
        .buf
        .copy_to(&mut got_bullish_reversal)?;
    result
        .outputs
        .bearish_reversal
        .buf
        .copy_to(&mut got_bearish_reversal)?;
    result
        .outputs
        .regular_bullish
        .buf
        .copy_to(&mut got_regular_bullish)?;
    result
        .outputs
        .regular_bearish
        .buf
        .copy_to(&mut got_regular_bearish)?;
    result
        .outputs
        .bullish_weakening
        .buf
        .copy_to(&mut got_bullish_weakening)?;
    result
        .outputs
        .bearish_weakening
        .buf
        .copy_to(&mut got_bearish_weakening)?;

    for idx in 0..cpu.oscillator.len() {
        assert!(
            approx_eq(cpu.oscillator[idx], got_oscillator[idx], 1e-6),
            "oscillator mismatch at {idx}: cpu={} cuda={}",
            cpu.oscillator[idx],
            got_oscillator[idx]
        );
        assert!(
            approx_eq(cpu.ma[idx], got_ma[idx], 1e-6),
            "ma mismatch at {idx}: cpu={} cuda={}",
            cpu.ma[idx],
            got_ma[idx]
        );
        assert!(
            approx_eq(cpu.bullish_reversal[idx], got_bullish_reversal[idx], 1e-9),
            "bullish_reversal mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_reversal[idx],
            got_bullish_reversal[idx]
        );
        assert!(
            approx_eq(cpu.bearish_reversal[idx], got_bearish_reversal[idx], 1e-9),
            "bearish_reversal mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_reversal[idx],
            got_bearish_reversal[idx]
        );
        assert!(
            approx_eq(cpu.regular_bullish[idx], got_regular_bullish[idx], 1e-9),
            "regular_bullish mismatch at {idx}: cpu={} cuda={}",
            cpu.regular_bullish[idx],
            got_regular_bullish[idx]
        );
        assert!(
            approx_eq(cpu.regular_bearish[idx], got_regular_bearish[idx], 1e-9),
            "regular_bearish mismatch at {idx}: cpu={} cuda={}",
            cpu.regular_bearish[idx],
            got_regular_bearish[idx]
        );
        assert!(
            approx_eq(cpu.bullish_weakening[idx], got_bullish_weakening[idx], 1e-9),
            "bullish_weakening mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_weakening[idx],
            got_bullish_weakening[idx]
        );
        assert!(
            approx_eq(cpu.bearish_weakening[idx], got_bearish_weakening[idx], 1e-9),
            "bearish_weakening mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_weakening[idx],
            got_bearish_weakening[idx]
        );
    }

    Ok(())
}
