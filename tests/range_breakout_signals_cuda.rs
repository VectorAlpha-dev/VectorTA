use vector_ta::indicators::range_breakout_signals::{
    range_breakout_signals_batch_with_kernel, RangeBreakoutSignalsBatchRange,
};
use vector_ta::utilities::enums::Kernel;

#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
use vector_ta::cuda::{cuda_available, CudaRangeBreakoutSignals};

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
fn range_breakout_signals_cuda_batch_matches_cpu() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_available() {
        eprintln!("[range_breakout_signals_cuda_batch_matches_cpu] skipped - no CUDA device");
        return Ok(());
    }

    let len = 1536usize;
    let mut open = vec![f64::NAN; len];
    let mut high = vec![f64::NAN; len];
    let mut low = vec![f64::NAN; len];
    let mut close = vec![f64::NAN; len];
    let mut volume = vec![f64::NAN; len];

    let mut base = 104.0f64;
    for i in 24..len {
        let x = i as f64;
        if i % 160 < 32 {
            base += (x * 0.009).sin() * 0.12;
        } else if i % 160 < 64 {
            base += 0.24 + (x * 0.015).sin() * 0.08;
        } else if i % 160 < 96 {
            base += (x * 0.011).cos() * 0.06;
        } else if i % 160 < 128 {
            base -= 0.27 + (x * 0.012).cos() * 0.09;
        } else {
            base += (x * 0.007).sin() * 0.05;
        }

        let spread = if i % 160 < 32 || (i % 160 >= 96 && i % 160 < 128) {
            0.10
        } else {
            0.72
        };
        let drift = if i % 160 < 32 {
            0.03
        } else if i % 160 < 64 {
            0.58
        } else if i % 160 < 96 {
            -0.04
        } else if i % 160 < 128 {
            -0.61
        } else {
            0.02
        };
        let o = base - drift * 0.5 + (x * 0.013).sin() * spread;
        let c = base + drift * 0.5 + (x * 0.017).cos() * spread;
        open[i] = o;
        close[i] = c;
        high[i] = o.max(c) + 0.24 + spread * 0.4;
        low[i] = o.min(c) - 0.23 - spread * 0.35;
        volume[i] = 1100.0
            + (i % 160) as f64 * 7.0
            + if drift > 0.4 || drift < -0.4 {
                850.0
            } else {
                240.0
            };
    }

    for i in (320..372).step_by(13) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }
    for i in (980..1040).step_by(11) {
        open[i] = f64::NAN;
        high[i] = f64::NAN;
        low[i] = f64::NAN;
        close[i] = f64::NAN;
        volume[i] = f64::NAN;
    }

    let sweep = RangeBreakoutSignalsBatchRange {
        range_length: (12, 18, 6),
        confirmation_length: (3, 5, 2),
    };
    let cpu = range_breakout_signals_batch_with_kernel(
        &open,
        &high,
        &low,
        &close,
        &volume,
        &sweep,
        Kernel::ScalarBatch,
    )?;
    let cuda = CudaRangeBreakoutSignals::new(0).expect("CudaRangeBreakoutSignals::new");
    let result = cuda
        .batch_dev(&open, &high, &low, &close, &volume, &sweep)
        .expect("batch_dev");

    assert_eq!(result.outputs.rows(), cpu.rows);
    assert_eq!(result.outputs.cols(), cpu.cols);
    assert_eq!(result.combos.len(), cpu.combos.len());

    let mut got_range_top = vec![0.0f64; result.outputs.range_top.len()];
    let mut got_range_bottom = vec![0.0f64; result.outputs.range_bottom.len()];
    let mut got_bullish = vec![0.0f64; result.outputs.bullish.len()];
    let mut got_extra_bullish = vec![0.0f64; result.outputs.extra_bullish.len()];
    let mut got_bearish = vec![0.0f64; result.outputs.bearish.len()];
    let mut got_extra_bearish = vec![0.0f64; result.outputs.extra_bearish.len()];
    result.outputs.range_top.buf.copy_to(&mut got_range_top)?;
    result
        .outputs
        .range_bottom
        .buf
        .copy_to(&mut got_range_bottom)?;
    result.outputs.bullish.buf.copy_to(&mut got_bullish)?;
    result
        .outputs
        .extra_bullish
        .buf
        .copy_to(&mut got_extra_bullish)?;
    result.outputs.bearish.buf.copy_to(&mut got_bearish)?;
    result
        .outputs
        .extra_bearish
        .buf
        .copy_to(&mut got_extra_bearish)?;

    for idx in 0..cpu.range_top.len() {
        assert!(
            approx_eq(cpu.range_top[idx], got_range_top[idx], 1e-9),
            "range_top mismatch at {idx}: cpu={} cuda={}",
            cpu.range_top[idx],
            got_range_top[idx]
        );
        assert!(
            approx_eq(cpu.range_bottom[idx], got_range_bottom[idx], 1e-9),
            "range_bottom mismatch at {idx}: cpu={} cuda={}",
            cpu.range_bottom[idx],
            got_range_bottom[idx]
        );
        assert!(
            approx_eq(cpu.bullish[idx], got_bullish[idx], 1e-9),
            "bullish mismatch at {idx}: cpu={} cuda={}",
            cpu.bullish[idx],
            got_bullish[idx]
        );
        assert!(
            approx_eq(cpu.extra_bullish[idx], got_extra_bullish[idx], 1e-9),
            "extra_bullish mismatch at {idx}: cpu={} cuda={}",
            cpu.extra_bullish[idx],
            got_extra_bullish[idx]
        );
        assert!(
            approx_eq(cpu.bearish[idx], got_bearish[idx], 1e-9),
            "bearish mismatch at {idx}: cpu={} cuda={}",
            cpu.bearish[idx],
            got_bearish[idx]
        );
        assert!(
            approx_eq(cpu.extra_bearish[idx], got_extra_bearish[idx], 1e-9),
            "extra_bearish mismatch at {idx}: cpu={} cuda={}",
            cpu.extra_bearish[idx],
            got_extra_bearish[idx]
        );
    }

    Ok(())
}
