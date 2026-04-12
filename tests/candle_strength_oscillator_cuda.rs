use vector_ta::indicators::candle_strength_oscillator::{
    candle_strength_oscillator_batch_with_kernel, CandleStrengthOscillatorBatchRange,
    CandleStrengthOscillatorParams,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaCandleStrengthOscillator};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let close: Vec<f64> = (0..len)
        .map(|i| {
            let x = i as f64;
            100.0 + x * 0.09 + (x * 0.13).sin() * 1.4 + (x * 0.031).cos() * 0.6
        })
        .collect();
    let open: Vec<f64> = close
        .iter()
        .enumerate()
        .map(|(i, &c)| c - ((i as f64) * 0.19).sin() * 0.7)
        .collect();
    let high: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .enumerate()
        .map(|(i, (&o, &c))| o.max(c) + 0.4 + ((i as f64) * 0.07).cos().abs() * 0.2)
        .collect();
    let low: Vec<f64> = open
        .iter()
        .zip(close.iter())
        .enumerate()
        .map(|(i, (&o, &c))| o.min(c) - 0.4 - ((i as f64) * 0.09).sin().abs() * 0.2)
        .collect();
    (open, high, low, close)
}

#[cfg(feature = "cuda")]
fn run_case(
    fixed: CandleStrengthOscillatorParams,
    sweep: CandleStrengthOscillatorBatchRange,
) -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[candle_strength_oscillator_cuda] skipped - no CUDA device");
        return Ok(());
    }

    let (mut open, mut high, mut low, mut close) = sample_ohlc(320);
    open[177] = f64::NAN;
    high[177] = f64::NAN;
    low[177] = f64::NAN;
    close[177] = f64::NAN;

    let cpu = candle_strength_oscillator_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &sweep,
        &fixed,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaCandleStrengthOscillator::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &sweep, &fixed)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_strength = vec![0.0f64; result.outputs.strength.len()];
    let mut got_highs = vec![0.0f64; result.outputs.highs.len()];
    let mut got_lows = vec![0.0f64; result.outputs.lows.len()];
    let mut got_mid = vec![0.0f64; result.outputs.mid.len()];
    let mut got_long_signal = vec![0.0f64; result.outputs.long_signal.len()];
    let mut got_short_signal = vec![0.0f64; result.outputs.short_signal.len()];
    result.outputs.strength.buf.copy_to(&mut got_strength)?;
    result.outputs.highs.buf.copy_to(&mut got_highs)?;
    result.outputs.lows.buf.copy_to(&mut got_lows)?;
    result.outputs.mid.buf.copy_to(&mut got_mid)?;
    result
        .outputs
        .long_signal
        .buf
        .copy_to(&mut got_long_signal)?;
    result
        .outputs
        .short_signal
        .buf
        .copy_to(&mut got_short_signal)?;

    for idx in 0..cpu.strength.len() {
        assert!(
            approx_eq(cpu.strength[idx], got_strength[idx], 1e-6),
            "strength mismatch at {idx}: cpu={} cuda={}",
            cpu.strength[idx],
            got_strength[idx]
        );
        assert!(
            approx_eq(cpu.highs[idx], got_highs[idx], 1e-6),
            "highs mismatch at {idx}: cpu={} cuda={}",
            cpu.highs[idx],
            got_highs[idx]
        );
        assert!(
            approx_eq(cpu.lows[idx], got_lows[idx], 1e-6),
            "lows mismatch at {idx}: cpu={} cuda={}",
            cpu.lows[idx],
            got_lows[idx]
        );
        assert!(
            approx_eq(cpu.mid[idx], got_mid[idx], 1e-6),
            "mid mismatch at {idx}: cpu={} cuda={}",
            cpu.mid[idx],
            got_mid[idx]
        );
        assert!(
            approx_eq(cpu.long_signal[idx], got_long_signal[idx], 1e-6),
            "long_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.long_signal[idx],
            got_long_signal[idx]
        );
        assert!(
            approx_eq(cpu.short_signal[idx], got_short_signal[idx], 1e-6),
            "short_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.short_signal[idx],
            got_short_signal[idx]
        );
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
fn candle_strength_oscillator_cuda_bollinger_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    run_case(
        CandleStrengthOscillatorParams {
            period: None,
            atr_enabled: Some(true),
            atr_length: None,
            mode: Some("bollinger".to_string()),
        },
        CandleStrengthOscillatorBatchRange {
            period: (28, 32, 4),
            atr_length: (40, 44, 4),
        },
    )
}

#[cfg(feature = "cuda")]
#[test]
fn candle_strength_oscillator_cuda_donchian_matches_cpu(
) -> Result<(), Box<dyn std::error::Error>> {
    run_case(
        CandleStrengthOscillatorParams {
            period: None,
            atr_enabled: Some(false),
            atr_length: None,
            mode: Some("donchian".to_string()),
        },
        CandleStrengthOscillatorBatchRange {
            period: (24, 28, 4),
            atr_length: (50, 50, 0),
        },
    )
}
