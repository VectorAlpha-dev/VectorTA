use vector_ta::indicators::trend_flow_trail::{
    trend_flow_trail_batch_with_kernel, TrendFlowTrailBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaTrendFlowTrail};

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        true
    } else {
        (a - b).abs() <= tol
    }
}

fn sample_ohlcv(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];
    let mut base = 101.0f64;
    for i in 18..len {
        let x = i as f64;
        base += (x * 0.008).sin() * 0.22 + (x * 0.0021).cos() * 0.07;
        let c = base + (x * 0.029).sin() * 1.18 + (x * 0.014).cos() * 0.41;
        let o = c - (x * 0.023).cos() * 0.47;
        let spread = 1.05 + (x * 0.017).sin().abs() * 0.36;
        open[i] = o;
        close[i] = c;
        high[i] = o.max(c) + spread;
        low[i] = o.min(c) - spread * (0.8 + (x * 0.011).cos().abs() * 0.22);
        volume[i] = 980.0 + x * 4.5 + (x * 0.043).sin() * 160.0 + (x * 0.009).cos() * 45.0;
    }
    (open, high, low, close, volume)
}

#[test]
fn cuda_feature_off_noop() {
    #[cfg(not(feature = "cuda"))]
    assert!(true);
}

#[cfg(feature = "cuda")]
#[test]
fn trend_flow_trail_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[trend_flow_trail_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let (mut open, mut high, mut low, mut close, mut volume) = sample_ohlcv(560);
    for i in 287..292 {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let sweep = TrendFlowTrailBatchRange {
        alpha_length: (29, 33, 4),
        alpha_multiplier: (2.9, 3.3, 0.4),
        mfi_length: (12, 14, 2),
    };

    let cpu = trend_flow_trail_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaTrendFlowTrail::new(0)?;
    let result = cuda.batch_dev(&open, &high, &low, &close, &volume, &sweep)?;

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.rows);

    let mut got_alpha_trail = vec![0.0f64; result.outputs.alpha_trail.len()];
    let mut got_alpha_trail_bullish = vec![0.0f64; result.outputs.alpha_trail_bullish.len()];
    let mut got_alpha_trail_bearish = vec![0.0f64; result.outputs.alpha_trail_bearish.len()];
    let mut got_alpha_dir = vec![0.0f64; result.outputs.alpha_dir.len()];
    let mut got_mfi = vec![0.0f64; result.outputs.mfi.len()];
    let mut got_tp_upper = vec![0.0f64; result.outputs.tp_upper.len()];
    let mut got_tp_lower = vec![0.0f64; result.outputs.tp_lower.len()];
    let mut got_alpha_trail_bullish_switch =
        vec![0.0f64; result.outputs.alpha_trail_bullish_switch.len()];
    let mut got_alpha_trail_bearish_switch =
        vec![0.0f64; result.outputs.alpha_trail_bearish_switch.len()];
    let mut got_mfi_overbought = vec![0.0f64; result.outputs.mfi_overbought.len()];
    let mut got_mfi_oversold = vec![0.0f64; result.outputs.mfi_oversold.len()];
    let mut got_mfi_cross_up_mid = vec![0.0f64; result.outputs.mfi_cross_up_mid.len()];
    let mut got_mfi_cross_down_mid = vec![0.0f64; result.outputs.mfi_cross_down_mid.len()];
    let mut got_price_cross_alpha_trail_up =
        vec![0.0f64; result.outputs.price_cross_alpha_trail_up.len()];
    let mut got_price_cross_alpha_trail_down =
        vec![0.0f64; result.outputs.price_cross_alpha_trail_down.len()];
    let mut got_mfi_above_90 = vec![0.0f64; result.outputs.mfi_above_90.len()];
    let mut got_mfi_below_10 = vec![0.0f64; result.outputs.mfi_below_10.len()];

    result
        .outputs
        .alpha_trail
        .buf
        .copy_to(&mut got_alpha_trail)?;
    result
        .outputs
        .alpha_trail_bullish
        .buf
        .copy_to(&mut got_alpha_trail_bullish)?;
    result
        .outputs
        .alpha_trail_bearish
        .buf
        .copy_to(&mut got_alpha_trail_bearish)?;
    result.outputs.alpha_dir.buf.copy_to(&mut got_alpha_dir)?;
    result.outputs.mfi.buf.copy_to(&mut got_mfi)?;
    result.outputs.tp_upper.buf.copy_to(&mut got_tp_upper)?;
    result.outputs.tp_lower.buf.copy_to(&mut got_tp_lower)?;
    result
        .outputs
        .alpha_trail_bullish_switch
        .buf
        .copy_to(&mut got_alpha_trail_bullish_switch)?;
    result
        .outputs
        .alpha_trail_bearish_switch
        .buf
        .copy_to(&mut got_alpha_trail_bearish_switch)?;
    result
        .outputs
        .mfi_overbought
        .buf
        .copy_to(&mut got_mfi_overbought)?;
    result
        .outputs
        .mfi_oversold
        .buf
        .copy_to(&mut got_mfi_oversold)?;
    result
        .outputs
        .mfi_cross_up_mid
        .buf
        .copy_to(&mut got_mfi_cross_up_mid)?;
    result
        .outputs
        .mfi_cross_down_mid
        .buf
        .copy_to(&mut got_mfi_cross_down_mid)?;
    result
        .outputs
        .price_cross_alpha_trail_up
        .buf
        .copy_to(&mut got_price_cross_alpha_trail_up)?;
    result
        .outputs
        .price_cross_alpha_trail_down
        .buf
        .copy_to(&mut got_price_cross_alpha_trail_down)?;
    result
        .outputs
        .mfi_above_90
        .buf
        .copy_to(&mut got_mfi_above_90)?;
    result
        .outputs
        .mfi_below_10
        .buf
        .copy_to(&mut got_mfi_below_10)?;

    for idx in 0..cpu.alpha_trail.len() {
        let row = idx / cpu.cols;
        let col = idx % cpu.cols;
        assert!(
            approx_eq(cpu.alpha_trail[idx], got_alpha_trail[idx], 1e-6),
            "alpha_trail mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.alpha_trail[idx],
            got_alpha_trail[idx]
        );
        assert!(
            approx_eq(
                cpu.alpha_trail_bullish[idx],
                got_alpha_trail_bullish[idx],
                1e-6
            ),
            "alpha_trail_bullish mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.alpha_trail_bullish[idx],
            got_alpha_trail_bullish[idx]
        );
        assert!(
            approx_eq(
                cpu.alpha_trail_bearish[idx],
                got_alpha_trail_bearish[idx],
                1e-6
            ),
            "alpha_trail_bearish mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.alpha_trail_bearish[idx],
            got_alpha_trail_bearish[idx]
        );
        assert!(
            approx_eq(cpu.alpha_dir[idx], got_alpha_dir[idx], 1e-6),
            "alpha_dir mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.alpha_dir[idx],
            got_alpha_dir[idx]
        );
        assert!(
            approx_eq(cpu.mfi[idx], got_mfi[idx], 1e-6),
            "mfi mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.mfi[idx],
            got_mfi[idx]
        );
        assert!(
            approx_eq(cpu.tp_upper[idx], got_tp_upper[idx], 1e-6),
            "tp_upper mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.tp_upper[idx],
            got_tp_upper[idx]
        );
        assert!(
            approx_eq(cpu.tp_lower[idx], got_tp_lower[idx], 1e-6),
            "tp_lower mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.tp_lower[idx],
            got_tp_lower[idx]
        );
        assert!(
            approx_eq(
                cpu.alpha_trail_bullish_switch[idx],
                got_alpha_trail_bullish_switch[idx],
                1e-6
            ),
            "alpha_trail_bullish_switch mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.alpha_trail_bullish_switch[idx],
            got_alpha_trail_bullish_switch[idx]
        );
        assert!(
            approx_eq(
                cpu.alpha_trail_bearish_switch[idx],
                got_alpha_trail_bearish_switch[idx],
                1e-6
            ),
            "alpha_trail_bearish_switch mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.alpha_trail_bearish_switch[idx],
            got_alpha_trail_bearish_switch[idx]
        );
        assert!(
            approx_eq(cpu.mfi_overbought[idx], got_mfi_overbought[idx], 1e-6),
            "mfi_overbought mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.mfi_overbought[idx],
            got_mfi_overbought[idx]
        );
        assert!(
            approx_eq(cpu.mfi_oversold[idx], got_mfi_oversold[idx], 1e-6),
            "mfi_oversold mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.mfi_oversold[idx],
            got_mfi_oversold[idx]
        );
        assert!(
            approx_eq(cpu.mfi_cross_up_mid[idx], got_mfi_cross_up_mid[idx], 1e-6),
            "mfi_cross_up_mid mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.mfi_cross_up_mid[idx],
            got_mfi_cross_up_mid[idx]
        );
        assert!(
            approx_eq(
                cpu.mfi_cross_down_mid[idx],
                got_mfi_cross_down_mid[idx],
                1e-6
            ),
            "mfi_cross_down_mid mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.mfi_cross_down_mid[idx],
            got_mfi_cross_down_mid[idx]
        );
        assert!(
            approx_eq(
                cpu.price_cross_alpha_trail_up[idx],
                got_price_cross_alpha_trail_up[idx],
                1e-6
            ),
            "price_cross_alpha_trail_up mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.price_cross_alpha_trail_up[idx],
            got_price_cross_alpha_trail_up[idx]
        );
        assert!(
            approx_eq(
                cpu.price_cross_alpha_trail_down[idx],
                got_price_cross_alpha_trail_down[idx],
                1e-6
            ),
            "price_cross_alpha_trail_down mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.price_cross_alpha_trail_down[idx],
            got_price_cross_alpha_trail_down[idx]
        );
        assert!(
            approx_eq(cpu.mfi_above_90[idx], got_mfi_above_90[idx], 1e-6),
            "mfi_above_90 mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.mfi_above_90[idx],
            got_mfi_above_90[idx]
        );
        assert!(
            approx_eq(cpu.mfi_below_10[idx], got_mfi_below_10[idx], 1e-6),
            "mfi_below_10 mismatch at row={row} col={col}: cpu={} cuda={}",
            cpu.mfi_below_10[idx],
            got_mfi_below_10[idx]
        );
    }

    Ok(())
}
