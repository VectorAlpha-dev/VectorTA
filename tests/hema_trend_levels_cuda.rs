use vector_ta::indicators::hema_trend_levels::{
    hema_trend_levels_batch_with_kernel, HemaTrendLevelsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaHemaTrendLevels};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlc(length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut open = Vec::with_capacity(length);
    let mut high = Vec::with_capacity(length);
    let mut low = Vec::with_capacity(length);
    let mut close = Vec::with_capacity(length);
    for i in 0..length {
        let x = i as f64;
        let o = if i < length / 3 {
            100.0 - x * 0.08 + (x * 0.03).sin() * 0.2
        } else if i < 2 * length / 3 {
            94.0 + x * 0.18 + (x * 0.05).sin() * 0.3
        } else {
            140.0 - x * 0.14 + (x * 0.04).cos() * 0.35
        };
        let c = o + (x * 0.07).cos() * 0.6;
        let h = o.max(c) + 0.75;
        let l = o.min(c) - 0.65;
        open.push(o);
        high.push(h);
        low.push(l);
        close.push(c);
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
fn hema_trend_levels_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[hema_trend_levels_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let (mut open, mut high, mut low, mut close) = sample_ohlc(360);
    open[201] = f64::NAN;
    high[201] = f64::NAN;
    low[201] = f64::NAN;
    close[201] = f64::NAN;

    let sweep = HemaTrendLevelsBatchRange {
        fast_length: (14, 18, 4),
        slow_length: (30, 34, 4),
    };

    let cpu =
        hema_trend_levels_batch_with_kernel(&open, &high, &low, &close, &sweep, Kernel::ScalarBatch)?;
    let cuda = CudaHemaTrendLevels::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_fast_hema = vec![0.0f64; result.outputs.fast_hema.len()];
    let mut got_slow_hema = vec![0.0f64; result.outputs.slow_hema.len()];
    let mut got_trend_direction = vec![0.0f64; result.outputs.trend_direction.len()];
    let mut got_bar_state = vec![0.0f64; result.outputs.bar_state.len()];
    let mut got_bullish_crossover = vec![0.0f64; result.outputs.bullish_crossover.len()];
    let mut got_bearish_crossunder = vec![0.0f64; result.outputs.bearish_crossunder.len()];
    let mut got_box_offset = vec![0.0f64; result.outputs.box_offset.len()];
    let mut got_bull_box_top = vec![0.0f64; result.outputs.bull_box_top.len()];
    let mut got_bull_box_bottom = vec![0.0f64; result.outputs.bull_box_bottom.len()];
    let mut got_bear_box_top = vec![0.0f64; result.outputs.bear_box_top.len()];
    let mut got_bear_box_bottom = vec![0.0f64; result.outputs.bear_box_bottom.len()];
    let mut got_bullish_test = vec![0.0f64; result.outputs.bullish_test.len()];
    let mut got_bearish_test = vec![0.0f64; result.outputs.bearish_test.len()];
    let mut got_bullish_test_level = vec![0.0f64; result.outputs.bullish_test_level.len()];
    let mut got_bearish_test_level = vec![0.0f64; result.outputs.bearish_test_level.len()];
    result.outputs.fast_hema.buf.copy_to(&mut got_fast_hema)?;
    result.outputs.slow_hema.buf.copy_to(&mut got_slow_hema)?;
    result
        .outputs
        .trend_direction
        .buf
        .copy_to(&mut got_trend_direction)?;
    result.outputs.bar_state.buf.copy_to(&mut got_bar_state)?;
    result
        .outputs
        .bullish_crossover
        .buf
        .copy_to(&mut got_bullish_crossover)?;
    result
        .outputs
        .bearish_crossunder
        .buf
        .copy_to(&mut got_bearish_crossunder)?;
    result.outputs.box_offset.buf.copy_to(&mut got_box_offset)?;
    result.outputs.bull_box_top.buf.copy_to(&mut got_bull_box_top)?;
    result
        .outputs
        .bull_box_bottom
        .buf
        .copy_to(&mut got_bull_box_bottom)?;
    result.outputs.bear_box_top.buf.copy_to(&mut got_bear_box_top)?;
    result
        .outputs
        .bear_box_bottom
        .buf
        .copy_to(&mut got_bear_box_bottom)?;
    result.outputs.bullish_test.buf.copy_to(&mut got_bullish_test)?;
    result.outputs.bearish_test.buf.copy_to(&mut got_bearish_test)?;
    result
        .outputs
        .bullish_test_level
        .buf
        .copy_to(&mut got_bullish_test_level)?;
    result
        .outputs
        .bearish_test_level
        .buf
        .copy_to(&mut got_bearish_test_level)?;

    for idx in 0..cpu.fast_hema.len() {
        assert!(
            approx_eq(cpu.fast_hema[idx], got_fast_hema[idx], 1e-6),
            "fast_hema mismatch at {idx}: cpu={} cuda={}",
            cpu.fast_hema[idx],
            got_fast_hema[idx]
        );
        assert!(
            approx_eq(cpu.slow_hema[idx], got_slow_hema[idx], 1e-6),
            "slow_hema mismatch at {idx}: cpu={} cuda={}",
            cpu.slow_hema[idx],
            got_slow_hema[idx]
        );
        assert!(
            approx_eq(cpu.trend_direction[idx], got_trend_direction[idx], 1e-6),
            "trend_direction mismatch at {idx}: cpu={} cuda={}",
            cpu.trend_direction[idx],
            got_trend_direction[idx]
        );
        assert!(
            approx_eq(cpu.bar_state[idx], got_bar_state[idx], 1e-6),
            "bar_state mismatch at {idx}: cpu={} cuda={}",
            cpu.bar_state[idx],
            got_bar_state[idx]
        );
        assert!(
            approx_eq(cpu.bullish_crossover[idx], got_bullish_crossover[idx], 1e-6),
            "bullish_crossover mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_crossover[idx],
            got_bullish_crossover[idx]
        );
        assert!(
            approx_eq(cpu.bearish_crossunder[idx], got_bearish_crossunder[idx], 1e-6),
            "bearish_crossunder mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_crossunder[idx],
            got_bearish_crossunder[idx]
        );
        assert!(
            approx_eq(cpu.box_offset[idx], got_box_offset[idx], 1e-6),
            "box_offset mismatch at {idx}: cpu={} cuda={}",
            cpu.box_offset[idx],
            got_box_offset[idx]
        );
        assert!(
            approx_eq(cpu.bull_box_top[idx], got_bull_box_top[idx], 1e-6),
            "bull_box_top mismatch at {idx}: cpu={} cuda={}",
            cpu.bull_box_top[idx],
            got_bull_box_top[idx]
        );
        assert!(
            approx_eq(cpu.bull_box_bottom[idx], got_bull_box_bottom[idx], 1e-6),
            "bull_box_bottom mismatch at {idx}: cpu={} cuda={}",
            cpu.bull_box_bottom[idx],
            got_bull_box_bottom[idx]
        );
        assert!(
            approx_eq(cpu.bear_box_top[idx], got_bear_box_top[idx], 1e-6),
            "bear_box_top mismatch at {idx}: cpu={} cuda={}",
            cpu.bear_box_top[idx],
            got_bear_box_top[idx]
        );
        assert!(
            approx_eq(cpu.bear_box_bottom[idx], got_bear_box_bottom[idx], 1e-6),
            "bear_box_bottom mismatch at {idx}: cpu={} cuda={}",
            cpu.bear_box_bottom[idx],
            got_bear_box_bottom[idx]
        );
        assert!(
            approx_eq(cpu.bullish_test[idx], got_bullish_test[idx], 1e-6),
            "bullish_test mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_test[idx],
            got_bullish_test[idx]
        );
        assert!(
            approx_eq(cpu.bearish_test[idx], got_bearish_test[idx], 1e-6),
            "bearish_test mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_test[idx],
            got_bearish_test[idx]
        );
        assert!(
            approx_eq(cpu.bullish_test_level[idx], got_bullish_test_level[idx], 1e-6),
            "bullish_test_level mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish_test_level[idx],
            got_bullish_test_level[idx]
        );
        assert!(
            approx_eq(cpu.bearish_test_level[idx], got_bearish_test_level[idx], 1e-6),
            "bearish_test_level mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish_test_level[idx],
            got_bearish_test_level[idx]
        );
    }

    Ok(())
}
