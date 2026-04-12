use vector_ta::indicators::bulls_v_bears::{
    bulls_v_bears_batch_with_kernel, BullsVBearsBatchRange, BullsVBearsCalculationMethod,
    BullsVBearsMaType,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaBullsVBears};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_hlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut base = 102.0f64;

    for i in 18..len {
        let x = i as f64;
        base += (x * 0.009).sin() * 0.33 + (x * 0.004).cos() * 0.12;
        let drift = if i % 180 < 40 {
            0.52
        } else if i % 180 < 92 {
            -0.48
        } else if i % 180 < 138 {
            0.17
        } else {
            -0.11
        };
        let c = base + drift + (x * 0.021).sin() * 0.74 + (x * 0.006).cos() * 0.29;
        let spread = 0.55 + (x * 0.013).sin().abs() * 0.28;
        high[i] = c + spread + 0.18;
        low[i] = c - spread - 0.16;
        close[i] = c;
    }

    for i in (260..330).step_by(11) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }
    for i in (910..980).step_by(13) {
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
    }

    (high, low, close)
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn bulls_v_bears_cuda_batch_normalized_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bulls_v_bears_cuda_batch_normalized_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let (high, low, close) = sample_hlc(1536);
    let sweep = BullsVBearsBatchRange {
        period: (10, 14, 4),
        normalized_bars_back: (80, 120, 40),
        raw_rolling_period: (50, 50, 0),
        raw_threshold_percentile: (95.0, 95.0, 0.0),
        threshold_level: (70.0, 90.0, 20.0),
        ma_type: BullsVBearsMaType::Ema,
        calculation_method: BullsVBearsCalculationMethod::Normalized,
    };
    let cpu = bulls_v_bears_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaBullsVBears::new(0).expect("CudaBullsVBears::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_value = vec![0.0f64; result.outputs.value.len()];
    let mut got_bull = vec![0.0f64; result.outputs.bull.len()];
    let mut got_bear = vec![0.0f64; result.outputs.bear.len()];
    let mut got_ma = vec![0.0f64; result.outputs.ma.len()];
    let mut got_upper = vec![0.0f64; result.outputs.upper.len()];
    let mut got_lower = vec![0.0f64; result.outputs.lower.len()];
    let mut got_bullish_signal = vec![0.0f64; result.outputs.bullish_signal.len()];
    let mut got_bearish_signal = vec![0.0f64; result.outputs.bearish_signal.len()];
    let mut got_zero_cross_up = vec![0.0f64; result.outputs.zero_cross_up.len()];
    let mut got_zero_cross_down = vec![0.0f64; result.outputs.zero_cross_down.len()];
    result.outputs.value.buf.copy_to(&mut got_value)?;
    result.outputs.bull.buf.copy_to(&mut got_bull)?;
    result.outputs.bear.buf.copy_to(&mut got_bear)?;
    result.outputs.ma.buf.copy_to(&mut got_ma)?;
    result.outputs.upper.buf.copy_to(&mut got_upper)?;
    result.outputs.lower.buf.copy_to(&mut got_lower)?;
    result
        .outputs
        .bullish_signal
        .buf
        .copy_to(&mut got_bullish_signal)?;
    result
        .outputs
        .bearish_signal
        .buf
        .copy_to(&mut got_bearish_signal)?;
    result
        .outputs
        .zero_cross_up
        .buf
        .copy_to(&mut got_zero_cross_up)?;
    result
        .outputs
        .zero_cross_down
        .buf
        .copy_to(&mut got_zero_cross_down)?;

    for idx in 0..cpu.value.len() {
        assert!(
            approx_eq(cpu.value[idx], got_value[idx], 1e-9),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.value[idx],
            got_value[idx]
        );
        assert!(
            approx_eq(cpu.bull[idx], got_bull[idx], 1e-9),
            "bull mismatch at {idx}: cpu={} cuda={}",
            cpu.bull[idx],
            got_bull[idx]
        );
        assert!(
            approx_eq(cpu.bear[idx], got_bear[idx], 1e-9),
            "bear mismatch at {idx}: cpu={} cuda={}",
            cpu.bear[idx],
            got_bear[idx]
        );
        assert!(
            approx_eq(cpu.ma[idx], got_ma[idx], 1e-9),
            "ma mismatch at {idx}: cpu={} cuda={}",
            cpu.ma[idx],
            got_ma[idx]
        );
        assert!(
            approx_eq(cpu.upper[idx], got_upper[idx], 1e-9),
            "upper mismatch at {idx}: cpu={} cuda={}",
            cpu.upper[idx],
            got_upper[idx]
        );
        assert!(
            approx_eq(cpu.lower[idx], got_lower[idx], 1e-9),
            "lower mismatch at {idx}: cpu={} cuda={}",
            cpu.lower[idx],
            got_lower[idx]
        );
        assert!(
            approx_eq(cpu.bullish_signal[idx], got_bullish_signal[idx], 1e-9),
            "bullish_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_signal[idx],
            got_bullish_signal[idx]
        );
        assert!(
            approx_eq(cpu.bearish_signal[idx], got_bearish_signal[idx], 1e-9),
            "bearish_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_signal[idx],
            got_bearish_signal[idx]
        );
        assert!(
            approx_eq(cpu.zero_cross_up[idx], got_zero_cross_up[idx], 1e-9),
            "zero_cross_up mismatch at {idx}: cpu={} cuda={}",
            cpu.zero_cross_up[idx],
            got_zero_cross_up[idx]
        );
        assert!(
            approx_eq(cpu.zero_cross_down[idx], got_zero_cross_down[idx], 1e-9),
            "zero_cross_down mismatch at {idx}: cpu={} cuda={}",
            cpu.zero_cross_down[idx],
            got_zero_cross_down[idx]
        );
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn bulls_v_bears_cuda_batch_raw_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[bulls_v_bears_cuda_batch_raw_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let (high, low, close) = sample_hlc(1536);
    let sweep = BullsVBearsBatchRange {
        period: (8, 12, 4),
        normalized_bars_back: (120, 120, 0),
        raw_rolling_period: (30, 50, 20),
        raw_threshold_percentile: (90.0, 95.0, 5.0),
        threshold_level: (80.0, 80.0, 0.0),
        ma_type: BullsVBearsMaType::Wma,
        calculation_method: BullsVBearsCalculationMethod::Raw,
    };
    let cpu = bulls_v_bears_batch_with_kernel(&high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaBullsVBears::new(0).expect("CudaBullsVBears::new");
    let result = cuda
        .batch_dev(&high, &low, &close, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_value = vec![0.0f64; result.outputs.value.len()];
    let mut got_bull = vec![0.0f64; result.outputs.bull.len()];
    let mut got_bear = vec![0.0f64; result.outputs.bear.len()];
    let mut got_ma = vec![0.0f64; result.outputs.ma.len()];
    let mut got_upper = vec![0.0f64; result.outputs.upper.len()];
    let mut got_lower = vec![0.0f64; result.outputs.lower.len()];
    let mut got_bullish_signal = vec![0.0f64; result.outputs.bullish_signal.len()];
    let mut got_bearish_signal = vec![0.0f64; result.outputs.bearish_signal.len()];
    let mut got_zero_cross_up = vec![0.0f64; result.outputs.zero_cross_up.len()];
    let mut got_zero_cross_down = vec![0.0f64; result.outputs.zero_cross_down.len()];
    result.outputs.value.buf.copy_to(&mut got_value)?;
    result.outputs.bull.buf.copy_to(&mut got_bull)?;
    result.outputs.bear.buf.copy_to(&mut got_bear)?;
    result.outputs.ma.buf.copy_to(&mut got_ma)?;
    result.outputs.upper.buf.copy_to(&mut got_upper)?;
    result.outputs.lower.buf.copy_to(&mut got_lower)?;
    result
        .outputs
        .bullish_signal
        .buf
        .copy_to(&mut got_bullish_signal)?;
    result
        .outputs
        .bearish_signal
        .buf
        .copy_to(&mut got_bearish_signal)?;
    result
        .outputs
        .zero_cross_up
        .buf
        .copy_to(&mut got_zero_cross_up)?;
    result
        .outputs
        .zero_cross_down
        .buf
        .copy_to(&mut got_zero_cross_down)?;

    for idx in 0..cpu.value.len() {
        assert!(
            approx_eq(cpu.value[idx], got_value[idx], 1e-9),
            "value mismatch at {idx}: cpu={} cuda={}",
            cpu.value[idx],
            got_value[idx]
        );
        assert!(
            approx_eq(cpu.bull[idx], got_bull[idx], 1e-9),
            "bull mismatch at {idx}: cpu={} cuda={}",
            cpu.bull[idx],
            got_bull[idx]
        );
        assert!(
            approx_eq(cpu.bear[idx], got_bear[idx], 1e-9),
            "bear mismatch at {idx}: cpu={} cuda={}",
            cpu.bear[idx],
            got_bear[idx]
        );
        assert!(
            approx_eq(cpu.ma[idx], got_ma[idx], 1e-9),
            "ma mismatch at {idx}: cpu={} cuda={}",
            cpu.ma[idx],
            got_ma[idx]
        );
        assert!(
            approx_eq(cpu.upper[idx], got_upper[idx], 1e-9),
            "upper mismatch at {idx}: cpu={} cuda={}",
            cpu.upper[idx],
            got_upper[idx]
        );
        assert!(
            approx_eq(cpu.lower[idx], got_lower[idx], 1e-9),
            "lower mismatch at {idx}: cpu={} cuda={}",
            cpu.lower[idx],
            got_lower[idx]
        );
        assert!(
            approx_eq(cpu.bullish_signal[idx], got_bullish_signal[idx], 1e-9),
            "bullish_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_signal[idx],
            got_bullish_signal[idx]
        );
        assert!(
            approx_eq(cpu.bearish_signal[idx], got_bearish_signal[idx], 1e-9),
            "bearish_signal mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_signal[idx],
            got_bearish_signal[idx]
        );
        assert!(
            approx_eq(cpu.zero_cross_up[idx], got_zero_cross_up[idx], 1e-9),
            "zero_cross_up mismatch at {idx}: cpu={} cuda={}",
            cpu.zero_cross_up[idx],
            got_zero_cross_up[idx]
        );
        assert!(
            approx_eq(cpu.zero_cross_down[idx], got_zero_cross_down[idx], 1e-9),
            "zero_cross_down mismatch at {idx}: cpu={} cuda={}",
            cpu.zero_cross_down[idx],
            got_zero_cross_down[idx]
        );
    }

    Ok(())
}
